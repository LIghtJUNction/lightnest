"""Bounded RV32 syntax graph and independent interpreter.

Adapted from the archived runtime/experiment/common.py. Only syntactic direct
relations are supplied to the model, never a return-ancestry mask or edits.
This shared runtime contains no neural architecture or model checkpoint.
"""
from collections import deque
from functools import lru_cache


def branch_offset(w):
    return sx(((w >> 31) << 12) | ((w >> 7 & 1) << 11) | ((w >> 25 & 63) << 5) | ((w >> 8 & 15) << 1), 13)


def jump_offset(w):
    return sx(((w >> 31) << 20) | ((w >> 12 & 255) << 12) | ((w >> 20 & 1) << 11) | ((w >> 21 & 1023) << 1), 21)


def s32(x): return (int(x)+2**31)%2**32-2**31

def sx(x,n): return (x+2**(n-1))%2**n-2**(n-1)

@lru_cache(maxsize=8192)
def graph(binary:bytes):
    """Static reaching definitions for registers and absolute scratch words.

    Direct edges only: no return reachability, target-operation whitelist, labels,
    simulator states, or correct edits. All outputs are compiler-style syntax.
    Computed load/store addresses and calls are explicitly unsupported.
    """
    if not binary or len(binary)%4 or len(binary)>1024: raise ValueError('bad code length')
    words=[int.from_bytes(binary[k:k+4],'little') for k in range(0,len(binary),4)];n=len(words)
    uses=[];writes=[];succ=[];pred=[[] for _ in words]
    for i,w in enumerate(words):
        op=w&127;rd=(w>>7)&31;a=(w>>15)&31;b=(w>>20)&31;f3=(w>>12)&7;f7=w>>25
        u=[];out=None;s=[i+1] if i+1<n else []
        if w==0x00100073:
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
            offset=branch_offset(w)
            if offset%4: raise ValueError('unaligned branch')
            u=[('r',a),('r',b)];s=[i+1,i+offset//4]
        elif op==0x6f and rd==0:
            offset=jump_offset(w)
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
