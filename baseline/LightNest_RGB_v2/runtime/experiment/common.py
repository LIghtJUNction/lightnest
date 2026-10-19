"""LightNest harder-task experiment. No labels or execution outcomes enter encoding."""
from __future__ import annotations
import sys, json, hashlib, gzip, io, importlib.util
from pathlib import Path
from functools import lru_cache
from collections import deque
import torch
ROOT=Path(__file__).resolve().parents[1]/'base'/'lightnest'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from lightnest import rv32
from lightnest.relation_patch import definition_use_graph, supervised_loss
from lightnest.growth_patch import GrowthConfig
ARCH_PATH=ROOT/'work/breakthrough-20261018-0029/arch/breakthrough_model.py'
spec=importlib.util.spec_from_file_location('harder_inherited_arch',ARCH_PATH)
arch=importlib.util.module_from_spec(spec);sys.modules[spec.name]=arch;spec.loader.exec_module(arch)
SEEDS=(104729,130363)
COMMANDS={0:'change every add that contributes to the output to sub',1:'change every sub that contributes to the output to add'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def s32(x): return (int(x)+2**31)%2**32-2**31
def sx(x,n): return (x+2**(n-1))%2**n-2**(n-1)

def parent_path(seed):
    m=json.loads((ROOT/'work/breakthrough-20261018-0029/accepted-parents.json').read_text())
    ent=next(x for x in m['models'] if x['seed']==seed)
    ref=ent['evaluation_spec']['checkpoint'];p=ROOT/ref['path']
    if sha(p)!=ref['sha256']: raise ValueError('parent checksum mismatch')
    return p

def load_parent(seed):
    with gzip.open(parent_path(seed),'rb') as f: payload=torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=True)
    m=arch.BreakthroughNetwork(GrowthConfig(**payload['config']),local_stop=False)
    m.load_state_dict(payload['state_dict'],strict=True)
    assert sum(p.numel() for p in m.parameters())==113138
    return m

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    m=arch.BreakthroughNetwork(GrowthConfig(**saved['config']),local_stop=False)
    m.load_state_dict(saved['state_dict'],strict=True);m.eval();return m

def assemble(source):
    """Existing assembler plus SW; labels retain the original raw-RV32 ABI."""
    labels={};placed=[];pc=256
    for line in source.splitlines():
        line=line.split('#',1)[0].strip()
        if not line: continue
        if line.endswith(':'):
            if line[:-1] in labels: raise ValueError('duplicate label')
            labels[line[:-1]]=pc
        else: placed.append((pc,line));pc+=4
    words=[]
    for pc,line in placed:
        parts=line.replace(',',' ').split()
        if parts[0]=='sw':
            imm,rs1=rv32._mem_operand(parts[2]);rs2=rv32._reg(parts[1]);rs1=rv32._reg(rs1)
            if not -2048<=imm<2048: raise ValueError('invalid store offset')
            v=imm&4095;word=((v>>5)<<25)|(rs2<<20)|(rs1<<15)|(2<<12)|((v&31)<<7)|0x23
        else: word=rv32._encode_line(line,pc,labels)
        words.append(word)
    return b''.join(w.to_bytes(4,'little') for w in words)

@lru_cache(maxsize=30000)
def graph(binary:bytes):
    """Static reaching definitions for registers and absolute scratch words.

    Direct edges only: no return reachability, target-operation whitelist, labels,
    simulator states, or correct edits. All outputs are compiler-style syntax.
    Computed load/store addresses and calls are explicitly unsupported.
    """
    if not binary or len(binary)%4 or len(binary)>1024: raise ValueError('bad code length')
    words=[int.from_bytes(binary[k:k+4],'little') for k in range(0,len(binary),4)];n=len(words)
    if not any(w&127 in (0x23,0x63,0x6f) for w in words):
        return definition_use_graph(binary).edges
    uses=[];writes=[];succ=[];pred=[[] for _ in words]
    for i,w in enumerate(words):
        op=w&127;rd=(w>>7)&31;a=(w>>15)&31;b=(w>>20)&31;f3=(w>>12)&7;f7=w>>25
        u=[];out=None;s=[i+1] if i+1<n else []
        if w==rv32.EBREAK:
            if i!=n-1: raise ValueError('early ebreak unsupported')
            u=[('r',10)];s=[]
        elif op==0x33:
            if not ((f7==0 and f3<=7) or (f7==32 and f3 in (0,5)) or (f7==1 and f3==0)): raise ValueError('bad R op')
            u=[('r',a),('r',b)];out=('r',rd) if rd else None
        elif op==0x13:
            if not (f3==0 or (f3==1 and f7==0) or (f3==5 and f7 in (0,32))): raise ValueError('bad I op')
            u=[('r',a)];out=('r',rd) if rd else None
        elif op==3 and f3==2:
            if a!=0: raise ValueError('computed memory alias unsupported')
            address=sx(w>>20,12)
            if address<0 or address>=256 or address%4: raise ValueError('memory address outside data area')
            u=[('m',address)];out=('r',rd) if rd else None
        elif op==0x23 and f3==2:
            if a!=0: raise ValueError('computed memory alias unsupported')
            address=sx(((w>>25)<<5)|((w>>7)&31),12)
            if not 64<=address<256 or address%4: raise ValueError('store outside scratch area')
            u=[('r',b)];out=('m',address)
        elif op==0x63 and f3 in (0,1,4):
            offset=rv32._b_imm(w)
            if offset%4: raise ValueError('unaligned branch')
            u=[('r',a),('r',b)];s=[i+1,i+offset//4]
        elif op==0x6f and rd==0:
            offset=rv32._j_imm(w)
            if offset%4: raise ValueError('unaligned jump')
            s=[i+offset//4]
        else: raise ValueError('unsupported control or memory instruction')
        if any(j<0 or j>=n for j in s): raise ValueError('branch leaves code')
        uses.append([k for k in u if k!=('r',0)]);writes.append(out);succ.append(s)
        for j in s: pred[j].append(i)
    # Reaching-definition fixed point, kills retain the latest write per path.
    after=[{} for _ in words];before=[{} for _ in words]
    q=deque(range(n));inq=set(q);rounds=0
    while q:
        i=q.popleft();inq.remove(i);rounds+=1
        if rounds>n*n*8: raise ValueError('reaching definitions did not converge')
        incoming={}
        for j in pred[i]:
            for key,val in after[j].items(): incoming[key]=incoming.get(key,frozenset())|val
        before[i]=incoming;new=dict(incoming)
        if writes[i] is not None:new[writes[i]]=frozenset((i,))
        if new!=after[i]:
            after[i]=new
            for j in succ[i]:
                if j not in inq:q.append(j);inq.add(j)
    edges={(i,j) for i in range(n) for key in uses[i] for j in before[i].get(key,()) if i!=j}
    # Branch control edges stop at the nearest common postdominating region.
    allnodes=set(range(n));pd=[set(allnodes) for _ in words];pd[-1]={n-1}
    for _ in range(n*4):
        changed=False
        for i in range(n-2,-1,-1):
            common=set.intersection(*(pd[j] for j in succ[i])) if succ[i] else set()
            new={i}|common
            if new!=pd[i]:pd[i]=new;changed=True
        if not changed:break
    for i,s in enumerate(succ):
        if len(s)!=2:continue
        stop=pd[i]-{i};todo=list(s);seen=set()
        while todo:
            j=todo.pop()
            if j in seen or j in stop or j==i:continue
            seen.add(j);edges.add((j,i));todo.extend(succ[j])
    return tuple(sorted(edges))


def encode(rows,config):
    raws=[r['command'].encode() for r in rows];codes=[bytes.fromhex(r['binary']) for r in rows]
    cw=max(map(len,raws));bw=max(map(len,codes));n=bw//4
    if cw>config.command_capacity or bw>config.binary_capacity:raise ValueError('capacity')
    c=torch.tensor([[v+1 for v in s]+[0]*(cw-len(s)) for s in raws],dtype=torch.long)
    b=torch.tensor([[v+1 for v in s]+[0]*(bw-len(s)) for s in codes],dtype=torch.long)
    rel=torch.zeros(len(rows),2,n,n,dtype=torch.float32)
    for k,code in enumerate(codes):
        ids=torch.arange(len(code)//4);rel[k,:,ids,ids]=1
        edges=graph(code)
        if edges:
            src,dst=zip(*edges);rel[k,0,list(src),list(dst)]=1;rel[k,1,list(dst),list(src)]=1
    rel/=rel.sum(-1,keepdim=True).clamp_min(1)
    return c,b,rel


def execute_a(binary,inputs,fuel=1800):
    """Original LightNest stepper with bounded data-only SW support."""
    import struct
    if not binary or len(binary)%4 or len(binary)>1024 or len(inputs)>16:raise ValueError('bad VM input')
    mem=bytearray(4096);mem[256:256+len(binary)]=binary
    for i,x in enumerate(inputs):struct.pack_into('<i',mem,4*i,s32(x))
    regs=[0]*32;regs[12]=len(inputs);pc=256
    for step in range(fuel):
        if pc<256 or pc>=256+len(binary) or pc%4:raise ValueError('PC outside code')
        w=int.from_bytes(mem[pc:pc+4],'little')
        if w==rv32.EBREAK:return regs[10]
        if w&127==0x23 and (w>>12)&7==2:
            a=(w>>15)&31;b=(w>>20)&31;address=regs[a]+sx(((w>>25)<<5)|((w>>7)&31),12)
            if not 64<=address<256 or address%4:raise ValueError('store outside scratch')
            struct.pack_into('<i',mem,address,s32(regs[b]));pc+=4
        else:
            pc=rv32._step(w,pc,regs,mem)
            if pc is None:raise ValueError('invalid instruction')
        regs[0]=0
    raise ValueError('fuel exhausted')


def execute_b(binary,inputs,fuel=1800):
    """Second decoder/VM, unsigned register storage; does not call rv32._step."""
    if not binary or len(binary)%4 or len(binary)>1024:raise ValueError('length')
    words=[int.from_bytes(binary[k:k+4],'little') for k in range(0,len(binary),4)]
    reg=[0]*32;reg[12]=len(inputs);memory={i*4:int(v)&0xffffffff for i,v in enumerate(inputs)};pc=0
    for _ in range(fuel):
        if not 0<=pc<len(words):raise ValueError('pc')
        w=words[pc];op=w&127;d=w>>7&31;f=w>>12&7;a=w>>15&31;b=w>>20&31;hi=w>>25;x=reg[a];y=reg[b];nxt=pc+1;v=None
        if w==0x00100073:return s32(reg[10])
        if op==0x33:
            if hi==0:
                if f==0:v=x+y
                elif f==1:v=x<<(y&31)
                elif f==2:v=int(s32(x)<s32(y))
                elif f==3:v=int(x<y)
                elif f==4:v=x^y
                elif f==5:v=x>>(y&31)
                elif f==6:v=x|y
                elif f==7:v=x&y
            elif hi==32 and f in (0,5):v=x-y if f==0 else s32(x)>>(y&31)
            elif hi==1 and f==0:v=x*y
            else:raise ValueError('R encoding')
        elif op==0x13:
            if f==0:v=x+sx(w>>20,12)
            elif f==1 and hi==0:v=x<<b
            elif f==5 and hi in (0,32):v=(s32(x) if hi else x)>>b
            else:raise ValueError('I encoding')
        elif op==3 and f==2:
            addr=s32(x)+sx(w>>20,12)
            if addr<0 or addr>=256 or addr%4:raise ValueError('load')
            v=memory.get(addr,0)
        elif op==0x23 and f==2:
            addr=s32(x)+sx(((w>>25)<<5)|(w>>7&31),12)
            if not 64<=addr<256 or addr%4:raise ValueError('store')
            memory[addr]=y
        elif op==0x63 and f in (0,1,4):
            imm=((w>>31)<<12)|((w>>7&1)<<11)|((w>>25&63)<<5)|((w>>8&15)<<1)
            take=(x==y if f==0 else x!=y if f==1 else s32(x)<s32(y))
            if take:
                off=sx(imm,13)
                if off%4:raise ValueError('branch')
                nxt=pc+off//4
        elif op==0x6f and d==0:
            imm=((w>>31)<<20)|((w>>12&255)<<12)|((w>>20&1)<<11)|((w>>21&1023)<<1)
            off=sx(imm,21)
            if off%4:raise ValueError('jump')
            nxt=pc+off//4
        else:raise ValueError('opcode')
        if v is not None and d:reg[d]=v&0xffffffff
        reg[0]=0;pc=nxt
    raise ValueError('fuel')
