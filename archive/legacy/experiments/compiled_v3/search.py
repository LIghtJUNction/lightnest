"""Budgeted one-instruction repair using supplied behavioral specifications.

No original bytes, source code, fault location, family, or hidden test is passed
into repair(). A passing sample is not a universal correctness certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import struct
from .vm import Input, Output, NativeVM, pack, MASK, signed

POLICIES=('all4','trace4','all64','trace64','trace_probe64')


def alternatives(w: int) -> list[int]:
    """Fixed typed search space; includes branch, arithmetic, width, and offset edits."""
    op=w&127;f=w>>12&7;h=w>>25;variants=set()
    if op==0x33:
        for ff,hh in ((0,0),(0,32),(0,1),(1,0),(2,0),(3,0),(4,0),(5,0),(5,32),(6,0),(7,0),(5,1),(7,1)):
            variants.add((w&~0xfe007000)|(hh<<25)|(ff<<12))
    elif op==0x63 and f in (0,1,4,5,6,7):
        variants.update((w&~0x7000)|(ff<<12) for ff in (0,1,4,5,6,7))
    elif op==0x13:
        if f in (0,2,3,4,6,7):
            variants.update((w&~0x7000)|(ff<<12) for ff in (0,2,3,4,6,7))
            immediate=signed(w>>20,12)
            for delta in (-4,-1,1,4):
                value=immediate+delta
                if -2048<=value<2048:variants.add((w&0xfffff)|((value&4095)<<20))
        elif f in (1,5) and h in (0,32):
            shift=w>>20&31
            for delta in (-1,1):variants.add((w&~(31<<20))|(((shift+delta)&31)<<20))
            if f==5:variants.add(w^(32<<25))
    elif op in (3,0x23):
        legal=(0,1,2,4,5) if op==3 else (0,1,2)
        if f in legal:variants.update((w&~0x7000)|(ff<<12) for ff in legal)
        im=signed(w>>20,12) if op==3 else signed(((w>>25)<<5)|(w>>7&31),12)
        for delta in (-4,4):
            value=im+delta
            if -2048<=value<2048:
                imm=value&4095
                variants.add((w&0xfffff)|(imm<<20) if op==3 else
                    (w&~0xfe000f80)|((imm>>5)<<25)|((imm&31)<<7))
    variants.discard(w)
    return sorted(variants)


def edits(code: bytes):
    for pos in range(0,len(code),4):
        old=int.from_bytes(code[pos:pos+4],'little')
        for word in alternatives(old):yield pos,word


def patched(code: bytes, pos: int, word: int) -> bytes:
    if pos%4 or not 0<=pos<=len(code)-4 or not 0<=word<=MASK:
        raise ValueError('Invalid instruction patch')
    return code[:pos]+struct.pack('<I',word)+code[pos+4:]


@dataclass(frozen=True)
class Request:
    code: bytes
    entry: int
    cases: tuple[Input,...]
    expected: tuple[Output,...]

    def __post_init__(self):
        if not 1<=len(self.cases)<=128 or len(self.cases)!=len(self.expected):
            raise ValueError('A bounded visible specification is required')
        if any(o.status for o in self.expected):raise ValueError('Reference tests must terminate')


@dataclass(frozen=True)
class Decision:
    code: bytes
    status: str
    patch: tuple[int,int] | None
    tried: int
    eligible: int
    vm_calls: int
    vm_steps: int
    assurance: str = 'sample_checked_not_proven'


def repair(request: Request, vm: NativeVM, *, guided: bool, budget=4096) -> Decision:
    """Propose one whole instruction edit or abstain. Never edit caller files."""
    if not 1<=budget<=100000:raise ValueError('Invalid search budget')
    before=tuple(vm.work);code=request.code;packed=pack(request.cases,request.expected)
    def result(status,change,tried,eligible):
        return Decision(code if change is None else patched(code,*change),status,change,tried,eligible,
                        int(vm.work[0]-before[0]),int(vm.work[1]-before[1]))
    first_failure=vm.check(code,request.entry,packed)
    if first_failure<0:return result('unchanged',None,0,0)
    coverage=None
    if guided:
        # A one-site fix must be executed OR read as code bytes on EACH original
        # failing input. This is necessary, not proof of semantic influence.
        hits=0
        for inp,expected in zip(request.cases[first_failure:],request.expected[first_failure:]):
            out,seen=vm.run(code,request.entry,inp,trace=True)
            if out.observable()!=expected.observable():
                coverage=seen if coverage is None else coverage&seen;hits+=1
                if hits==4:break
    candidates=list(edits(code))
    candidates.sort(key=lambda pair:hashlib.sha256(code+struct.pack('<II',*pair)).digest())
    if coverage is not None:candidates=[p for p in candidates if p[0]//4 in coverage]
    tried=0
    for change in candidates[:budget]:
        tried+=1
        if vm.check(patched(code,*change),request.entry,packed)<0:
            return result('candidate',change,tried,len(candidates))
    return result('budget' if len(candidates)>budget else 'abstain',None,tried,len(candidates))


def literal_values(code: bytes) -> list[int]:
    """Conservative constant *probe suggestions*, not a static correctness proof.

    Scans LUI/ADDI patterns and immediate fields; ignores path feasibility.
    Wrong suggestions cost tests, never justify accepting a patch.
    """
    regs={0:0};values=[]
    for pos in range(0,len(code),4):
        w=int.from_bytes(code[pos:pos+4],'little');op=w&127;d=w>>7&31;a=w>>15&31;f=w>>12&7
        value=None
        if op==0x37:value=w&0xfffff000
        elif op==0x13 and f==0 and a in regs:value=(regs[a]+signed(w>>20,12))&MASK
        if value is not None:values.append(value)
        if op in (0x37,0x17,0x13,0x33,3,0x6f,0x67) and d:
            if value is None:regs.pop(d,None)
            else:regs[d]=value
    # Complete constants before aligned LUI prefixes. Large literals reach rare
    # dispatch paths that uniform random 32-bit keys will usually miss.
    unique=list(dict.fromkeys(v for v in values if v>65535))
    unique.sort(key=lambda v:(v&4095==0,))
    return unique
