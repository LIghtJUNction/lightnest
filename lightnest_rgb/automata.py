"""Explicit finite automata, L* observation-table learner, exact product checker.

Learner receives membership labels and external counterexamples. It never reads
the reference transitions. Exact equivalence is relative to the supplied DFA,
not a formal proof of the CPython implementation or emitted native code.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class DFA:
    transition: np.ndarray
    accept: np.ndarray
    start: int=0
    def __post_init__(self):
        self.transition=np.asarray(self.transition);self.accept=np.asarray(self.accept)
        n=len(self.accept)
        if self.transition.shape!=(n,256) or not 1<=n<=255:raise ValueError('1..255 states, 256 symbols required')
        if not np.issubdtype(self.transition.dtype,np.integer) or (self.transition<0).any() or (self.transition>=n).any():raise ValueError('invalid transition')
        if not np.isin(self.accept,(0,1)).all() or not 0<=self.start<n:raise ValueError('invalid outputs/start')
        self.transition=self.transition.astype(np.uint8);self.accept=self.accept.astype(np.uint8)
    def run(self,data:bytes)->int:
        q=self.start
        for a in data:q=int(self.transition[q,a])
        return int(self.accept[q])
    def json(self):return dict(transition=self.transition.tolist(),accept=self.accept.tolist(),start=self.start)


def utf8_reference()->DFA:
    """Manually transcribed ABNF from RFC3629 section 4, not used by learner."""
    t=np.full((9,256),8,np.uint8)
    t[0,:128]=0;t[0,0xc2:0xe0]=1
    t[0,0xe0]=4;t[0,0xe1:0xed]=2;t[0,0xed]=5;t[0,0xee:0xf0]=2
    t[0,0xf0]=6;t[0,0xf1:0xf4]=3;t[0,0xf4]=7
    t[1,0x80:0xc0]=0;t[2,0x80:0xc0]=1;t[3,0x80:0xc0]=2
    t[4,0xa0:0xc0]=1;t[5,0x80:0xa0]=1;t[6,0x90:0xc0]=2;t[7,0x80:0x90]=2
    a=np.zeros(9,np.uint8);a[0]=1
    return DFA(t,a)


def equivalent(ref:DFA,cand:DFA):
    """BFS reachable product. Return relation on success, shortest witness on failure."""
    start=(ref.start,cand.start);todo=deque([start]);parent={start:None};edges=0
    while todo:
        s,t=todo.popleft()
        if ref.accept[s]!=cand.accept[t]:
            seq=[];q=(s,t)
            while parent[q] is not None:
                prev,a=parent[q];seq.append(a);q=prev
            return dict(equivalent=False,witness_hex=bytes(seq[::-1]).hex(),visited_pairs=len(parent),checked_edges=edges)
        for a in range(256):
            pair=(int(ref.transition[s,a]),int(cand.transition[t,a]));edges+=1
            if pair not in parent:parent[pair]=((s,t),a);todo.append(pair)
    return dict(equivalent=True,witness_hex=None,visited_pairs=len(parent),checked_edges=edges,relation=[list(x) for x in parent])


def verify_relation(ref:DFA,cand:DFA,relation):
    """Independent direct closure check; does not call BFS equivalence."""
    pairs={tuple(x) for x in relation}
    if (ref.start,cand.start) not in pairs:return False
    for s,t in pairs:
        if not 0<=s<len(ref.accept) or not 0<=t<len(cand.accept):return False
        if ref.accept[s]!=cand.accept[t]:return False
        for a in range(256):
            if (int(ref.transition[s,a]),int(cand.transition[t,a])) not in pairs:return False
    return True


class TableLearner:
    """Direct L* implementation. External exact equivalence is a strong teacher."""
    def __init__(self,member,equiv):
        self.member=member;self.equiv=equiv;self.cache={};self.S=[b''];self.E=[b''];self.log=[]
        self.rowcache={};self.hypotheses=[]
    def ask(self,w):
        if w not in self.cache:self.cache[w]=int(bool(self.member(w)))
        return self.cache[w]
    def row(self,s):
        if s not in self.rowcache:self.rowcache[s]=tuple(self.ask(s+e) for e in self.E)
        return self.rowcache[s]
    def close_consistent(self):
        alphabet=[bytes([a]) for a in range(256)]
        for _ in range(4096):
            representatives={self.row(s):s for s in self.S};added=False
            for s in self.S:
                for a in alphabet:
                    if self.row(s+a) not in representatives:
                        self.S.append(s+a);added=True;break
                if added:break
            if added:continue
            # Equal observed rows must have equal successor rows.
            bad=None
            for i,s in enumerate(self.S):
                for t in self.S[i+1:]:
                    if self.row(s)!=self.row(t):continue
                    for a in alphabet:
                        rs,rt=self.row(s+a),self.row(t+a)
                        if rs!=rt:
                            j=next(j for j in range(len(self.E)) if rs[j]!=rt[j]);bad=a+self.E[j];break
                    if bad is not None:break
                if bad is not None:break
            if bad is not None:
                if bad in self.E:raise AssertionError('duplicate distinguishing suffix')
                self.E.append(bad);self.rowcache={};continue
            return
        raise RuntimeError('closure budget exhausted')
    def hypothesis(self):
        reps={}
        for s in self.S:
            if self.row(s) not in reps:reps[self.row(s)]=s
        rows=list(reps);ids={r:i for i,r in enumerate(rows)}
        tr=np.array([[ids[self.row(reps[r]+bytes([a]))] for a in range(256)] for r in rows],np.uint8)
        acc=np.array([self.ask(reps[r]) for r in rows],np.uint8)
        return DFA(tr,acc,ids[self.row(b'')])
    def learn(self,max_equivalence_queries=32):
        for iteration in range(max_equivalence_queries):
            self.close_consistent();cand=self.hypothesis();self.hypotheses.append(cand)
            result=self.equiv(cand)
            record=dict(iteration=iteration,states=len(cand.accept),unique_membership_queries=len(self.cache),
                        suffixes=len(self.E),prefixes=len(self.S),**result)
            self.log.append(record)
            if result['equivalent']:return cand
            ce=bytes.fromhex(result['witness_hex'])
            if cand.run(ce)==self.ask(ce):raise AssertionError('counterexample disagrees with membership oracle')
            for i in range(len(ce)+1):
                if ce[:i] not in self.S:self.S.append(ce[:i])
        raise RuntimeError('equivalence budget exhausted')


def emit_dfa(machine:DFA)->bytes:
    """Trusted x86-64 table interpreter; input RDI, length RSI, returns Boolean RAX.

    No compilation after generation; instruction skeleton supplied, not learned.
    Every input load is preceded by an unsigned index/length comparison.
    """
    code=bytearray();labels={};fixes=[]
    def emit(h):code.extend(bytes.fromhex(h))
    def rel(op,name):
        emit(op);fixes.append((len(code),name));code.extend(b'\x00'*4)
    emit('b8');code.extend(int(machine.start).to_bytes(4,'little'));emit('31c9')
    rel('4c8d05','table')
    labels['loop']=len(code);emit('4839f1');rel('0f83','done')
    emit('0fb6140f c1e008 01d0 410fb60400 48ffc1');rel('e9','loop')
    labels['done']=len(code);rel('4c8d05','accept');emit('410fb60400 c3')
    labels['table']=len(code);code.extend(machine.transition.tobytes())
    labels['accept']=len(code);code.extend(machine.accept.tobytes())
    for off,name in fixes:code[off:off+4]=int(labels[name]-off-4).to_bytes(4,'little',signed=True)
    if len(code)>4096:raise ValueError('bounded native runner supports code <=4096 bytes')
    return bytes(code)


def emit_demo_elf(function:bytes,data:bytes)->bytes:
    """Only trusted encoder output allowed. Embedded input, writes uint64 stdout."""
    import struct
    if len(data)>131072:raise ValueError('demo input too large')
    stub=bytearray.fromhex('488d3d00000000be00000000e80000000050')
    stub+=bytes.fromhex('b801000000bf010000004889e6ba080000000f05b83c00000031ff0f05')
    stub[8:12]=struct.pack('<I',len(data));stub[13:17]=struct.pack('<i',len(stub)-17)
    stub[3:7]=struct.pack('<i',len(stub)+len(function)-7)
    payload=bytes(stub)+function+data;base=0x400000;off=120
    ident=b'\x7fELF'+bytes([2,1,1,0])+bytes(8)
    header=struct.pack('<16sHHIQQQIHHHHHH',ident,2,62,1,base+off,64,0,0,64,56,1,0,0,0)
    seg=struct.pack('<IIQQQQQQ',1,5,0,base,base,off+len(payload),off+len(payload),4096)
    return header+seg+payload
