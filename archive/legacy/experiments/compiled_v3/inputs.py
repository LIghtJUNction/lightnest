"""Deterministic behavioral tests; targeted values are derived only from input code."""
from __future__ import annotations
import hashlib
import random
from .vm import Input, MASK
from .search import literal_values


def identity(inp: Input) -> str:
    return hashlib.sha256(bytes(inp)).hexdigest()


def random_inputs(seed: int, count: int) -> list[Input]:
    rng=random.Random(seed);result=[]
    for i in range(count):
        kind=i%8
        if kind==0:a=[rng.randrange(16) for _ in range(32)]
        elif kind==1:a=[rng.getrandbits(32)]*32
        elif kind==2:a=sorted(rng.getrandbits(32) for _ in range(32))
        elif kind==3:a=sorted((rng.getrandbits(32) for _ in range(32)),reverse=True)
        elif kind==4:a=[rng.randrange(48,58) for _ in range(32)]
        elif kind==5:a=[rng.choice([0,1,31,32,127,255,65535,0x7fffffff,0x80000000,MASK]) for _ in range(32)]
        elif kind==6:a=[rng.randrange(256) for _ in range(32)]
        else:a=[rng.getrandbits(32) for _ in range(32)]
        n=[0,1,32,16][i] if i<4 else rng.choice([0,1,2,15,16,31,32,rng.randrange(33)])
        key=rng.choice([0,1,31,32,MASK,0x80000000,rng.getrandbits(32)])
        result.append(Input.make(a,n,key,rng.randrange(32)))
    return result


def probe_inputs(code: bytes, seed: int, count=64) -> list[Input]:
    if count<16:raise ValueError('Probe budget must be at least 16')
    initial=random_inputs(seed,16);pool=random_inputs(seed^0x38a4c53, max(count*4,256))
    literals=literal_values(code) or [0,1,32,MASK]
    targets=[]
    for value in literals:
        for delta in (0,-1,1):
            for mode in ('key','array'):
                targets.append(((value+delta)&MASK,mode))
    for i in range(count-16):
        inp=pool[i];value,mode=targets[i%len(targets)]
        inp.n=32 if i%3!=2 else 1
        if mode=='key':inp.key=value
        else:inp.a[0]=value
        initial.append(inp)
    return initial


def hidden_inputs(code: bytes, seed: int, forbidden: set[str], count=256) -> list[Input]:
    # Different seed, data, and observations from both public suites. Include
    # branches triggered by compiled constants as a preregistered stress stratum.
    candidates=random_inputs(seed,count)+probe_inputs(code,seed^0x42ff067, max(count,64))
    out=[];seen=set(forbidden)
    order=[]
    for i in range(count):order.extend((candidates[i],candidates[count+i]))
    for inp in order:
        key=identity(inp)
        if key not in seen:out.append(inp);seen.add(key)
        if len(out)==count:return out
    raise ValueError('Cannot form disjoint hidden set')
