"""Original harder-task generator, retained for deterministic new-source generation.
These are synthetic instruction-transform tasks, not historical software bugs.
"""
import random
from .vm import assemble
COMMANDS={0:'change every add that contributes to the output to sub',1:'change every sub that contributes to the output to add'}
FAMILIES=('mixed','overwrite','spill','branch','loop')

class Builder:
    def __init__(self,rng,direction):
        self.rng=rng;self.source_op='add' if direction==0 else 'sub'
        self.lines=[];self.index=0;self.reg={};self.mem={};self.label_count=0
    def label(self,name):self.lines.append(name+':')
    def raw(self,text):self.lines.append(text);i=self.index;self.index+=1;return i
    def op(self,mn,rd,a,b=None):
        da=set(self.reg.get(a,()))
        if mn in ('addi','slli','srli','srai'):i=self.raw(f'{mn} x{rd}, x{a}, {b}')
        else:
            da |= self.reg.get(b,set());i=self.raw(f'{mn} x{rd}, x{a}, x{b}')
        if mn==self.source_op: da.add(i)
        if rd:self.reg[rd]=da
        return i
    def load(self,rd,address):
        self.raw(f'lw x{rd}, {address}(x0)');self.reg[rd]=set(self.mem.get(address,()))
    def store(self,rs,address):
        self.raw(f'sw x{rs}, {address}(x0)');self.mem[address]=set(self.reg.get(rs,()))
    def noise(self,regs,count):
        for _ in range(count):self.op('addi',self.rng.choice(regs),0,self.rng.randint(-1900,1900))
    def chain(self,r,y,z,length,max_targets=3):
        nt=0
        for k in range(length):
            if k==0 or (nt<max_targets and self.rng.random()<.3):mn=self.source_op;nt+=1
            else:mn=self.rng.choice(['xor','mul','or','and','slli','srli','srai','addi'])
            if mn in ('slli','srli','srai'):self.op(mn,r,r,self.rng.randint(1,3))
            elif mn=='addi':self.op(mn,r,r,self.rng.randint(-15,15))
            else:self.op(mn,r,r,self.rng.choice((y,z)))
    def arm_branch(self,r,base,x,y,z,length):
        tag=f'b{self.label_count}';self.label_count+=1
        self.raw(f'blt x{x}, x{y}, {tag}_left');before={k:set(v) for k,v in self.reg.items()}
        self.op('addi',r,base,0);self.chain(r,y,z,length,max_targets=2);right=set(self.reg[r])
        self.raw(f'jal x0, {tag}_end');self.label(tag+'_left');self.reg=before
        self.op('addi',r,base,0);self.chain(r,z,y,length+1,max_targets=2);left=set(self.reg[r])
        self.label(tag+'_end');self.reg[r]=left|right
    def arm_loop(self,r,base,y,z,counter,length):
        tag=f'l{self.label_count}';self.label_count+=1
        self.op('addi',r,base,0);self.op('addi',counter,0,self.rng.randint(2,5));self.label(tag)
        self.chain(r,y,z,length,max_targets=2)
        self.op('addi',counter,counter,-1);self.raw(f'bne x{counter}, x0, {tag}')


def generate_pair(family,seed,direction,length_range=(2,5),prefix=''):
    rng=random.Random(seed);g=Builder(rng,direction)
    rs=rng.sample([i for i in range(1,32) if i!=10],17)
    x,y,z,base,tmp,ra,rb,counta,countb,dead0,dead1,spare,*noise=rs
    for r,addr in ((x,0),(y,4),(z,8)):g.load(r,addr)
    g.noise(noise,rng.randint(0,5))
    g.op('addi',tmp,x,rng.randint(1,19));g.op('xor',base,y,z);g.op('mul',base,base,tmp)
    for lane,r,cnt in ((0,ra,counta),(1,rb,countb)):
        length=rng.randint(*length_range)
        if family.startswith('branch'):g.arm_branch(r,base,x,y,z,length)
        elif family.startswith('loop'):g.arm_loop(r,base,y,z,cnt,length)
        else:
            g.op('addi',r,base,0)
            if family=='overwrite':
                g.chain(r,y,z,2,max_targets=1);g.op('addi',dead0,r,0)
                g.op('xor',dead0,dead0,z);g.op('addi',r,base,0)
            g.chain(r,y,z,length,max_targets=3)
        if 'spill' in family:
            addr=64+lane*16
            g.op(g.source_op,dead1,y,z);g.store(dead1,addr)
            g.store(r,addr);g.op('addi',r,0,rng.randint(1,99));g.load(r,addr);g.op('xor',r,r,z)
            g.store(dead1,addr+4)
        g.noise(noise,rng.randint(1,5));g.op(g.source_op,dead0,x,y);g.op('mul',dead1,dead0,z)
    rows=[];root=f'{prefix}/{family}/{seed}'
    for lane,r in enumerate((ra,rb)):
        source='\n'.join(g.lines+[f'addi x10, x{r}, 0','ebreak'])+'\n'
        original=assemble(source);target=bytearray(original);edits=[]
        for instruction in sorted(g.reg[r]):
            p=4*instruction+3;new=original[p]^64;word=int.from_bytes(original[p-3:p+1],'little')
            assert word&127==51 and (word>>12)&7==0 and word>>25==(0 if direction==0 else 32)
            target[p]=new;edits.append({'offset':p,'value':new})
        assert 1<=len(edits)<=8,(family,len(edits));target=bytes(target)
        rows.append(dict(id=f'{root}/return{lane}/edit',pair_id=root,family=family,direction=direction,lane=lane,
            command=COMMANDS[direction],binary=original.hex(),target=target.hex(),edits=edits,source=source))
        rows.append(dict(id=f'{root}/return{lane}/noop',pair_id=root,family=family,direction=direction,lane=lane,
            command=COMMANDS[direction],binary=target.hex(),target=target.hex(),edits=[],source=None))
    left=bytes.fromhex(rows[0]['binary']);right=bytes.fromhex(rows[2]['binary'])
    assert len(left)==len(right) and 1<=sum(a!=b for a,b in zip(left,right))<=2
    assert all(i in (len(left)-7,len(left)-6) for i,(a,b) in enumerate(zip(left,right)) if a!=b)
    return rows


def states(episodes):
    rows=[]
    for ep in episodes:
        edits=ep['edits'];n=len(edits)
        if not n:continue
        masks=range(1<<n) if n<=3 else sorted({0,(1<<n)-1,*((1<<k)-1 for k in range(1,n)),*(1<<k for k in range(n))})
        for mask in masks:
            b=bytearray.fromhex(ep['binary']);remaining=[]
            for i,e in enumerate(edits):
                if mask>>i&1:b[e['offset']]=e['value']
                else:remaining.append(e)
            rows.append({k:v for k,v in ep.items() if k!='source'}|dict(binary=bytes(b).hex(),edits=remaining,
                id=ep['id']+f'/state{mask}',parent_id=ep['id']))
    return rows


