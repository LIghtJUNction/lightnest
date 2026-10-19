"""Lossless template and delta I/O. Templates never authorize output reuse.

This is a bounded research codec, not a replacement for VCDIFF/Zstandard.
The RV32 key is for compression only. It MUST NOT key neural states or patches.
"""
from __future__ import annotations
import hashlib
import io
import json
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable

MAX_BLOB = 16 * 1024 * 1024

def uint(n: int) -> bytes:
    if type(n) is not int or not 0 <= n < (1 << 63):
        raise ValueError('integer out of range')
    out=bytearray()
    while n >= 128:
        out.append((n & 127) | 128); n >>= 7
    out.append(n)
    return bytes(out)

def read_uint(f: io.BytesIO, maximum: int=(1 << 63)-1) -> int:
    n=0
    for shift in range(0,63,7):
        b=f.read(1)
        if not b: raise ValueError('truncated integer')
        v=b[0];n |= (v & 127) << shift
        if not v & 128:
            if n > maximum: raise ValueError('length or identifier exceeds limit')
            return n
    raise ValueError('oversized integer')

def take(f: io.BytesIO, n: int) -> bytes:
    if not 0 <= n <= MAX_BLOB: raise ValueError('blob too large')
    b=f.read(n)
    if len(b)!=n: raise ValueError('truncated blob')
    return b

def blob(b: bytes) -> bytes:
    if len(b)>MAX_BLOB:raise ValueError('blob too large')
    return uint(len(b))+b

def read_blob(f: io.BytesIO) -> bytes:
    return take(f,read_uint(f,MAX_BLOB))

def make_delta(base: bytes, target: bytes) -> bytes:
    """Same-length overwrites with literal fallback; never guesses missing data."""
    if max(len(base),len(target))>MAX_BLOB:raise ValueError('blob too large')
    if base==target:return b'\x00'
    literal=b'\x02'+blob(target)
    if len(base)!=len(target):return literal
    out=bytearray(b'\x01');runs=[];i=0
    while i<len(base):
        if base[i]==target[i]:i+=1;continue
        start=i
        while i<len(base) and base[i]!=target[i]:i+=1
        runs.append((start,target[start:i]))
    out+=uint(len(runs));last=0
    for off,data in runs:
        out+=uint(off-last)+blob(data);last=off+len(data)
    return bytes(out) if len(out)<len(literal) else literal

def apply_delta(base: bytes, encoded: bytes) -> bytes:
    f=io.BytesIO(encoded);tag=f.read(1)
    if tag==b'\x00':out=base
    elif tag==b'\x02':out=read_blob(f)
    elif tag==b'\x01':
        out=bytearray(base);last=0
        for _ in range(read_uint(f,MAX_BLOB)):
            off=last+read_uint(f,MAX_BLOB);data=read_blob(f)
            if not data or off<last or off+len(data)>len(base):raise ValueError('patch outside base or empty edit')
            out[off:off+len(data)]=data;last=off+len(data)
        out=bytes(out)
    else:raise ValueError('unknown delta type')
    if f.read(1):raise ValueError('trailing bytes')
    return out

def compression_template_key(program: bytes) -> bytes:
    """Operand-independent key; all removed bits are preserved by the delta.

    ADD and SUB are grouped for compression. This key deliberately does not
    establish equivalent behavior, graph topology, or valid neural cache reuse.
    Unknown instructions are kept unchanged. Different sizes never share a key.
    """
    if len(program)%4:return b'RAW'+hashlib.sha256(program).digest()
    words=[]
    for (w,) in struct.iter_unpack('<I',program):
        op=w&127
        if op==0x33:
            w &= 0xfe00707f # funct7, funct3, opcode
            if w in (0x33,0x40000033):w=0x33
        elif op in (0x13,0x03,0x67):
            f3=(w>>12)&7
            w &= 0xfe00707f if op==0x13 and f3 in (1,5) else 0x707f
        elif op in (0x23,0x63):w &= 0x707f
        elif op in (0x37,0x17,0x6f):w=op
        words.append(w)
    return b'RV32'+struct.pack('<'+'I'*len(words),*words)

@dataclass(frozen=True)
class CoreRow:
    record_id: bytes
    command: str
    binary: bytes
    target: bytes

def pack_core(rows: Iterable[CoreRow], templates: bool) -> tuple[bytes,dict]:
    body=io.BytesIO();body.write(b'LNTP1' if templates else b'LNRW1')
    command_ids={};keys={};bases=[];count=0;dbytes=0;changed=0;raw_bytes=0
    for row in rows:
        if len(row.record_id)!=32:raise ValueError('record id must be SHA256')
        if row.command not in command_ids:
            command_ids[row.command]=len(command_ids)
            body.write(b'C'+blob(row.command.encode('utf8')))
        if templates:
            key=compression_template_key(row.binary)
            if key not in keys:
                keys[key]=len(bases);bases.append(row.binary);body.write(b'T'+blob(row.binary))
            tid=keys[key];din=make_delta(bases[tid],row.binary);dout=make_delta(row.binary,row.target)
            body.write(b'R'+row.record_id+uint(command_ids[row.command])+uint(tid)+blob(din)+blob(dout))
            dbytes+=len(dout)
        else:
            body.write(b'R'+row.record_id+uint(command_ids[row.command])+blob(row.binary)+blob(row.target))
        count+=1;raw_bytes+=len(row.binary)+len(row.target)
        changed+=sum(a!=b for a,b in zip(row.binary,row.target))+abs(len(row.binary)-len(row.target))
    payload=body.getvalue();packed=payload+hashlib.sha256(payload).digest()
    return packed,dict(records=count,templates=len(bases),template_raw_bytes=sum(map(len,bases)),commands=len(command_ids),
        input_plus_output_raw_bytes=raw_bytes,output_delta_bytes=dbytes,changed_output_bytes=changed,packed_bytes=len(packed))

def unpack_core(packed: bytes, max_records: int=2_000_000) -> Iterable[CoreRow]:
    if len(packed)<37 or hashlib.sha256(packed[:-32]).digest()!=packed[-32:]:raise ValueError('archive hash mismatch')
    f=io.BytesIO(packed[:-32]);magic=f.read(5)
    if magic not in (b'LNTP1',b'LNRW1'):raise ValueError('format version')
    commands=[];bases=[];count=0
    while (tag:=f.read(1)):
        if tag==b'C':commands.append(read_blob(f).decode('utf8'))
        elif tag==b'T' and magic==b'LNTP1':bases.append(read_blob(f))
        elif tag==b'R':
            rid=take(f,32);cid=read_uint(f)
            if cid>=len(commands):raise ValueError('unknown command')
            if magic==b'LNTP1':
                tid=read_uint(f)
                if tid>=len(bases):raise ValueError('missing template')
                b=apply_delta(bases[tid],read_blob(f));t=apply_delta(b,read_blob(f))
            else:b=read_blob(f);t=read_blob(f)
            count+=1
            if count>max_records:raise ValueError('too many records')
            yield CoreRow(rid,commands[cid],b,t)
        else:raise ValueError('unknown record tag')

@dataclass(frozen=True)
class CacheContext:
    """Every semantic/version boundary is part of the exact cache key."""
    tenant: str
    model_sha256: str
    runtime_sha256: str
    architecture: str='rv32im_restricted'
    abi: str='lightnest-rv32-scratch-v1'
    rounds: int=64
    max_edits: int=8
    decoding: str='deterministic-argmax-fp32-cpu-v1'

class ExactRequestCache:
    """Bounded, process-local cache, only for frozen deterministic inference.

    The full command and full input bytes are verified as well as the hash.
    Entries preserve errors and incorrect predictions; this is not validation.
    No API is provided for reuse across training weight updates.
    """
    def __init__(self,max_entries: int=512,max_bytes: int=16<<20):
        if max_entries<1 or max_bytes<1:raise ValueError('cache bounds')
        self.max_entries=max_entries;self.max_bytes=max_bytes;self.bytes=0
        self.data=OrderedDict();self.hits=0;self.misses=0
    def clear(self):self.data.clear();self.bytes=0
    def get_or_compute(self,ctx: CacheContext,command: str,program: bytes,compute: Callable[[],bytes]) -> bytes:
        scope=json.dumps(ctx.__dict__,sort_keys=True,separators=(',',':')).encode()
        signature=(scope,command.encode(),program)
        h=hashlib.sha256(blob(scope)+blob(signature[1])+blob(program)).digest()
        entry=self.data.get(h)
        if entry is not None and entry[0]==signature:
            self.hits+=1;self.data.move_to_end(h);return entry[1]
        self.misses+=1;value=compute()
        if not isinstance(value,bytes):raise TypeError('cache output must be immutable bytes')
        size=sum(map(len,signature))+len(value)+32
        if size<=self.max_bytes:
            if h in self.data:self.bytes-=self.data.pop(h)[2]
            while self.data and (len(self.data)>=self.max_entries or self.bytes+size>self.max_bytes):
                self.bytes-=self.data.popitem(last=False)[1][2]
            self.data[h]=(signature,value,size);self.bytes+=size
        return value

def transport_packet(base: bytes,target: bytes) -> bytes:
    """Bind a patch to exact base and target hashes, never to a fuzzy template."""
    return b'LNP1'+hashlib.sha256(base).digest()+hashlib.sha256(target).digest()+blob(make_delta(base,target))

def restore_packet(base: bytes,packet: bytes) -> bytes:
    f=io.BytesIO(packet)
    if take(f,4)!=b'LNP1':raise ValueError('packet version')
    if take(f,32)!=hashlib.sha256(base).digest():raise ValueError('wrong base')
    target_hash=take(f,32);target=apply_delta(base,read_blob(f))
    if f.read(1):raise ValueError('packet trailing bytes')
    if target_hash!=hashlib.sha256(target).digest():raise ValueError('wrong target')
    return target

_GEAR=tuple(int.from_bytes(hashlib.sha256(('LightNest-gear-v1:'+str(i)).encode()).digest()[:8],'little') for i in range(256))

def chunks(data: bytes,kind: str='gear',size: int=4096):
    """Bounded educational Gear CDC; NOT an implementation of FastCDC's tuning."""
    if size<16 or size&(size-1):raise ValueError('chunk target must be a power of two >=16')
    if kind=='fixed':
        for start in range(0,len(data),size):yield start,data[start:start+size]
        return
    if kind!='gear':raise ValueError('chunk mode')
    start=0
    while start<len(data):
        i=min(len(data),start+size//4);end=min(len(data),start+size*4);h=0
        while i<end:
            h=((h<<1)+_GEAR[data[i]])&((1<<64)-1);i+=1
            if not h&(size-1):break
        yield start,data[start:i];start=i

def chunk_packet(base: bytes,target: bytes,kind: str='gear',size: int=4096) -> tuple[bytes,dict]:
    table={}
    for off,b in chunks(base,kind,size):table.setdefault(hashlib.sha256(b).digest(),[]).append((off,len(b)))
    out=bytearray(b'LNC1'+hashlib.sha256(base).digest()+hashlib.sha256(target).digest()+uint(len(target)))
    copies=0;literals=0;count=0
    for _,b in chunks(target,kind,size):
        choices=table.get(hashlib.sha256(b).digest(),())
        off=next((off for off,n in choices if n==len(b) and base[off:off+n]==b),None)
        if off is None:out+=b'A'+blob(b);literals+=len(b)
        else:out+=b'C'+uint(off)+uint(len(b));copies+=len(b)
        count+=1
    return bytes(out),dict(reused_bytes=copies,literal_bytes=literals,chunks=count,packet_bytes=len(out),kind=kind,target_chunk_bytes=size)

def restore_chunks(base: bytes,packet: bytes,max_output: int=MAX_BLOB) -> bytes:
    f=io.BytesIO(packet)
    if take(f,4)!=b'LNC1':raise ValueError('chunk packet version')
    if take(f,32)!=hashlib.sha256(base).digest():raise ValueError('chunk base mismatch')
    target_hash=take(f,32);n=read_uint(f,max_output);out=bytearray()
    while (tag:=f.read(1)):
        if tag==b'C':
            off=read_uint(f,len(base));length=read_uint(f,len(base))
            if off+length>len(base):raise ValueError('copy out of bounds')
            data=base[off:off+length]
        elif tag==b'A':data=read_blob(f)
        else:raise ValueError('unknown copy opcode')
        if len(out)+len(data)>n:raise ValueError('output limit')
        out+=data
    if len(out)!=n or hashlib.sha256(out).digest()!=target_hash:raise ValueError('chunk target mismatch')
    return bytes(out)
