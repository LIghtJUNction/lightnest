"""Universal instruction/behavior encoder and learned byte edit-set decoder."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .targets import canonical_arch, decode, graph, token

ARCHITECTURE="lightnest-universal-machine-v1"


@dataclass(frozen=True)
class Config:
    width: int=64
    rounds: int=8
    heads: int=4
    max_bytes: int=4096
    max_instructions: int=768
    instruction_slots: int=16
    max_edits: int=64
    trained_architectures: tuple[str,...]=("x86_64",)

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name!="trained_architectures" and (type(value)is not int or value<=0):
                raise ValueError(f"{name} must be a positive integer")
        if self.width%4 or self.width%self.heads or self.max_bytes>16384 or self.max_instructions>1024 or self.instruction_slots>32 or self.rounds>64:
            raise ValueError("model configuration exceeds capacity")
        if not self.trained_architectures or any(canonical_arch(a)!=a for a in self.trained_architectures):
            raise ValueError("trained architecture names must be canonical")


def validate(row,config):
    if set(row)!={"architecture","command","binary","examples"}:
        raise ValueError("inference accepts architecture, command, binary and visible examples only")
    if not isinstance(row['architecture'],str) or not isinstance(row['command'],str) or not 1<=len(row['command'].encode())<=384:
        raise ValueError("invalid architecture or command")
    if not isinstance(row['binary'],str):raise ValueError("binary must be hex text")
    code=bytes.fromhex(row['binary'])
    if code.hex()!=row['binary'].lower() or not 1<=len(code)<=config.max_bytes:raise ValueError("invalid code bytes")
    instructions=decode(canonical_arch(row['architecture']),code)
    if len(instructions)>config.max_instructions or any(len(i.raw)>config.instruction_slots for i in instructions):
        raise ValueError("decoded block exceeds model instruction capacity")
    examples=row['examples']
    if not isinstance(examples,list) or not 1<=len(examples)<=16:raise ValueError("need 1..16 visible examples")
    for example in examples:
        if not isinstance(example,dict) or set(example)!={'inputs','output'} or not isinstance(example['inputs'],list) or len(example['inputs'])!=3:
            raise ValueError("visible example requires three uint64 inputs and one uint64 output")
        if any(type(x)is not int or not 0<=x<2**64 for x in [*example['inputs'],example['output']]):
            raise ValueError("visible example values must be uint64")
    return code,instructions


@dataclass
class Batch:
    commands: torch.Tensor
    raw: torch.Tensor
    fields: torch.Tensor
    valid: torch.Tensor
    relations: torch.Tensor
    offsets: torch.Tensor
    examples: torch.Tensor
    example_valid: torch.Tensor
    architectures: torch.Tensor
    codes: list[bytes]

    @classmethod
    def encode(cls,rows,config,device='cpu'):
        if not rows:raise ValueError("empty batch")
        decoded=[validate(r,config) for r in rows];b=len(rows);n=max(len(x[1]) for x in decoded);s=config.instruction_slots
        commands=torch.zeros(b,max(len(r['command'].encode()) for r in rows),dtype=torch.long)
        raw=torch.zeros(b,n,s,dtype=torch.long);fields=torch.zeros(b,n,6,dtype=torch.long)
        valid=torch.zeros(b,n,dtype=torch.bool);rel=torch.zeros(b,2,n,n)
        offsets=torch.zeros(b,n,s,dtype=torch.long)
        examples=torch.zeros(b,max(len(r['examples']) for r in rows),32,dtype=torch.long)
        ev=torch.zeros(examples.shape[:2],dtype=torch.bool);arches=torch.zeros(b,dtype=torch.long)
        for j,(row,(code,instructions)) in enumerate(zip(rows,decoded,strict=True)):
            text=list(row['command'].encode());commands[j,:len(text)]=torch.tensor(text)+1
            arches[j]=token(canonical_arch(row['architecture']),64)
            for i,ins in enumerate(instructions):
                size=len(ins.raw);raw[j,i,:size]=torch.tensor(list(ins.raw))+1;valid[j,i]=True
                offsets[j,i,:size]=torch.arange(ins.offset,ins.offset+size)
                fields[j,i]=torch.tensor([ins.mnemonic%2048,ins.flow%16,size,
                    int(ins.memory_read)+2*int(ins.memory_write),
                    token(ins.reads[0],256) if ins.reads else 0,token(ins.writes[0],256) if ins.writes else 0])
            ids=torch.arange(len(instructions));rel[j,:,ids,ids]=1
            for user,definition in graph(instructions):rel[j,0,user,definition]=1;rel[j,1,definition,user]=1
            for k,e in enumerate(row['examples']):
                numbers=[*e['inputs'],e['output']]
                examples[j,k]=torch.tensor([((v>>(8*i))&255)+1 for v in numbers for i in range(8)])
                ev[j,k]=True
        rel/=rel.sum(-1,keepdim=True).clamp_min(1)
        tensors=[commands,raw,fields,valid,rel,offsets,examples,ev,arches]
        return cls(*(t.to(device) for t in tensors),[x[0] for x in decoded])


class MachineNetwork(nn.Module):
    def __init__(self,config=Config()):
        super().__init__();self.config=config;d=config.width
        self.byte=nn.Embedding(257,d//4,padding_idx=0)
        self.slot=nn.Embedding(config.instruction_slots,d//4)
        self.command=nn.Embedding(257,d,padding_idx=0)
        self.command_conv=nn.Conv1d(d,d,5,padding=2)
        self.example=nn.Sequential(nn.Linear(32*d//4,d),nn.GELU(),nn.Linear(d,d))
        self.arch=nn.Embedding(64,d)
        self.fields=nn.ModuleList(nn.Embedding(size,d//4) for size in (2048,16,33,4,256,256))
        self.input=nn.Sequential(nn.Linear(config.instruction_slots*d//4+6*d//4,d),nn.LayerNorm(d),nn.GELU())
        self.before=nn.TransformerEncoderLayer(d,config.heads,2*d,dropout=0,batch_first=True,norm_first=True)
        self.message=nn.Linear(3*d,d);self.cell=nn.GRUCell(d,d);self.norm=nn.LayerNorm(d)
        self.after=nn.TransformerEncoderLayer(d,config.heads,2*d,dropout=0,batch_first=True,norm_first=True)
        self.readout=nn.Sequential(nn.Linear(3*d+d//4,d),nn.GELU(),nn.LayerNorm(d))
        self.edit=nn.Linear(d,1);self.bits=nn.Linear(d,8)

    def forward(self,batch):
        c,raw,valid=batch.commands,batch.raw,batch.valid;d=self.config.width
        cm=c.ne(0);command=F.gelu(self.command_conv(self.command(c).transpose(1,2))).transpose(1,2)
        command=(command*cm[...,None]).sum(1)/cm.sum(1,keepdim=True)
        example=self.example(self.byte(batch.examples).flatten(-2))
        behavior=(example*batch.example_valid[...,None]).sum(1)/batch.example_valid.sum(1,keepdim=True)
        condition=command+behavior+self.arch(batch.architectures)
        byte=self.byte(raw)
        h=self.input(torch.cat((byte.flatten(-2),*(emb(batch.fields[...,k]) for k,emb in enumerate(self.fields))),-1))
        positions=torch.arange(h.shape[1],device=h.device,dtype=h.dtype)[:,None]
        freq=torch.exp(torch.arange(0,d,2,device=h.device,dtype=h.dtype)*(-math.log(10000)/d))
        pos=torch.stack(((positions*freq).sin(),(positions*freq).cos()),-1).flatten(-2)
        inputs=h+condition[:,None]+pos
        h=self.before(inputs,src_key_padding_mask=~valid).masked_fill(~valid[...,None],0)
        for _ in range(self.config.rounds):
            message=self.message(torch.cat((inputs,torch.bmm(batch.relations[:,0],h),torch.bmm(batch.relations[:,1],h)),-1))
            h=self.norm(self.cell(message.flatten(0,1),h.flatten(0,1)).reshape_as(h)).masked_fill(~valid[...,None],0)
        h=self.after(h,src_key_padding_mask=~valid).masked_fill(~valid[...,None],0)
        pooled=h.sum(1)/valid.sum(1,keepdim=True)
        ctx=torch.cat((h,pooled[:,None].expand_as(h),condition[:,None].expand_as(h)),-1)
        ctx=ctx[:,:,None].expand(-1,-1,self.config.instruction_slots,-1)
        decoded=self.readout(torch.cat((ctx,byte+self.slot(torch.arange(self.config.instruction_slots,device=h.device))),-1))
        slot_valid=raw.ne(0);length=max(map(len,batch.codes));indices=batch.offsets.flatten(1)
        edit=self.edit(decoded).squeeze(-1).masked_fill(~slot_valid,0).flatten(1)
        bits=self.bits(decoded).masked_fill(~slot_valid[...,None],0).flatten(1,2)
        logits=torch.zeros(len(h),length,device=h.device).scatter_add(1,indices,edit)
        replacements=torch.zeros(len(h),length,8,device=h.device).scatter_add(1,indices[...,None].expand_as(bits),bits)
        mask=torch.arange(length,device=h.device)[None]<torch.tensor(list(map(len,batch.codes)),device=h.device)[:,None]
        return logits.masked_fill(~mask,-torch.inf),replacements

    @torch.inference_mode()
    def predict(self,rows):
        if self.training:raise RuntimeError("eval() required")
        if any(canonical_arch(r.get('architecture','')) not in self.config.trained_architectures for r in rows):
            raise ValueError("this checkpoint has no training evidence for the requested architecture")
        batch=Batch.encode(rows,self.config,next(self.parameters()).device);logits,bits=self(batch)
        outputs=[]
        for j,code in enumerate(batch.codes):
            z=logits[j,:len(code)];v=bits[j,:len(code)]
            if not torch.isfinite(z).all() or not torch.isfinite(v).all():raise FloatingPointError("nonfinite prediction")
            patches=[{'offset':p,'value':sum(int(x>=0)<<k for k,x in enumerate(v[p]))} for p in (z>0).nonzero().flatten().tolist()]
            status='candidate'
            if len(patches)>self.config.max_edits:status='budget'
            elif any(code[p['offset']]==p['value'] for p in patches):status='no_progress'
            final=bytearray(code)
            if status=='candidate':
                for patch in patches:final[patch['offset']]=patch['value']
                try:decode(rows[j]['architecture'],bytes(final))
                except ValueError:status='invalid_encoding';final=bytearray(code)
            outputs.append({'architecture':canonical_arch(rows[j]['architecture']),'patches':patches,'status':status,
                            'final':bytes(final).hex(),'stop':status=='candidate','assurance':'unverified_neural_candidate'})
        return outputs


def loss_for(logits,bits,batch,rows):
    target=torch.zeros_like(logits);values=torch.zeros_like(bits)
    valid=torch.zeros_like(logits,dtype=torch.bool)
    if len(rows)!=len(batch.codes):raise ValueError("training batch mismatch")
    for j,(code,row) in enumerate(zip(batch.codes,rows,strict=True)):
        final=bytes.fromhex(row['target'])
        if len(final)!=len(code):raise ValueError("current decoder trains fixed-length substitutions")
        valid[j,:len(code)]=True
        for p,(a,b) in enumerate(zip(code,final,strict=True)):
            if a!=b:
                target[j,p]=1;values[j,p]=torch.tensor([(b>>k)&1 for k in range(8)],device=bits.device)
    pos=target.bool();neg=valid&~pos
    losses=F.binary_cross_entropy_with_logits(logits.masked_fill(~valid,0),target,reduction='none')
    location=((losses*pos).sum(1)/pos.sum(1).clamp_min(1)+(losses*neg).sum(1)/neg.sum(1).clamp_min(1)+losses.masked_fill(~neg,0).max(1).values).mean()
    replacement=F.binary_cross_entropy_with_logits(bits[pos],values[pos]) if pos.any() else bits.sum()*0
    return location+2*replacement


def save(path,model,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save({'architecture':ARCHITECTURE,'config':asdict(model.config),'state_dict':model.state_dict(),'metadata':metadata},path)


def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['architecture']!=ARCHITECTURE:raise ValueError("checkpoint architecture mismatch")
    config=payload['config'];config['trained_architectures']=tuple(config['trained_architectures'])
    model=MachineNetwork(Config(**config));model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval()
