"""Hybrid latent recurrence and structured action emission, not a general LLM.

The linked state (not a decoded thought string) is fed into the next recurrent
block. Every discrete emission is reversible structured text. Mode decisions
are sampled only during learning; latent transitions are deterministic.
A binary edit invalidates the graph/state: subsequent proposals re-encode it.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,math,re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import ROOT,BASE
from rgb.network import load_rgb,RGBNetwork
from rgb.runtime import common,configure_torch
from rgb.scripted_core import LoopCore
from study.evaluate import action
DEPTHS=(4,8,16,32,64)
ARCH='lightnest-hybrid-state-action-v1'
FEATURES=197

class Link(nn.Module):
    def __init__(self,d=64,rank=32):
        super().__init__();self.norm=nn.LayerNorm(d)
        self.down=nn.Linear(2*d,rank);self.up=nn.Linear(rank,d)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,h,g,valid):
        y=h+self.up(F.gelu(self.down(torch.cat((self.norm(h),g[:,None].expand_as(h)),-1))))
        return y.masked_fill(~valid[...,None],0.)

class ActionHead(nn.Module):
    def __init__(self,red):
        super().__init__()
        for name in ('byte_fusion','pointer','replacement','stop_fusion','stop'):
            setattr(self,name,deepcopy(getattr(red,name)))
        self.none_index=red.none_index;self.local_stop_enabled=False
        for p in self.parameters():p.requires_grad_(True)
    def forward(self,h,ctx):
        return common.arch.BreakthroughNetwork.decode_state(self,h,ctx)

class ModePolicy(nn.Module):
    """0 = continue latent work, 1 = emit an action. Never sees labels."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,2))
        self.register_buffer('mean',torch.zeros(FEATURES));self.register_buffer('std',torch.ones(FEATURES))
    def normalize_from(self,x):
        self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.03))
    def forward(self,x):return self.net(((x-self.mean)/self.std).clamp(-20,20))

class HybridNetwork(nn.Module):
    def __init__(self,seed):
        super().__init__();rgb=load_rgb(BASE/f'models/rgb-{seed}.pt')
        self.red=rgb.red;self.green=rgb.green
        for p in self.red.parameters():p.requires_grad_(False)
        for p in self.green.parameters():p.requires_grad_(False)
        self.head=ActionHead(self.red);self.link=Link();self.seed=seed
        # Reuse the audited same-math frozen loop. TorchScript is an optional,
        # deprecated upstream CPU tool, not a new learned architecture.
        object.__setattr__(self,'loop',torch.jit.script(LoopCore(self.red).eval()))
    @property
    def none_index(self):return self.red.none_index
    def train(self,mode=True):
        super().train(mode);self.red.eval();self.green.eval();return self
    def start(self,rows):
        # The strict input surface excludes any answers and group metadata.
        for r in rows:
            if set(r)!={'command','binary'}:raise ValueError('inference accepts command and binary only')
        c,b,rel=common.encode(rows,self.red.config)
        with torch.no_grad():inputs,rel,ctx=self.red.encode_inputs(c,b,rel)
        return inputs,rel,ctx,inputs
    def block(self,h,inputs,rel,ctx,steps,linked=True):
        h=self.loop(h,inputs,rel,ctx.valid,steps)
        g=self.green(h,ctx.command,ctx.valid).slots.mean(1)
        if linked:h=self.link(h,g,ctx.valid)
        if not bool(torch.isfinite(h).all()):raise FloatingPointError('nonfinite hybrid state')
        return h,g
    def feature(self,h,g,ctx,scores,depth):
        p=scores.softmax(-1);top=p.topk(2,-1).values
        mean=h.sum(1)/ctx.valid.sum(1,keepdim=True)
        x=torch.cat((mean,g,ctx.command,top[:,:1],top[:,:1]-top[:,1:],p[:,-1:],
              torch.full_like(p[:,:1],depth/64),ctx.valid.sum(1,keepdim=True).to(h.dtype).log()/8),-1)
        if x.shape[-1]!=FEATURES or not bool(torch.isfinite(x).all()):raise FloatingPointError('feature')
        return x
    def full(self,rows,*,linked=True,teacher=False):
        inputs,rel,ctx,h=self.start(rows);original=h;out=[];previous=0
        for d in DEPTHS:
            h,g=self.block(h,inputs,rel,ctx,d-previous,linked);sc,bi=self.head(h,ctx)
            item={'h':h,'scores':sc,'bits':bi,'features':self.feature(h,g,ctx,sc,d),'depth':d,'valid':ctx.valid}
            if teacher:
                with torch.no_grad():original=self.loop(original,inputs,rel,ctx.valid,d-previous)
                item['teacher_h']=original.detach()
            out.append(item);previous=d
        return out
    @torch.inference_mode()
    def predict(self,rows,policy=None,*,fixed64=False,linked=True):
        if self.training:raise RuntimeError('eval required')
        inputs,rel,ctx,h=self.start(rows);ids=torch.arange(len(rows));out=[None]*len(rows);traces=[[] for _ in rows];prev=0
        for d in DEPTHS:
            h,g=self.block(h,inputs,rel,ctx,d-prev,linked);sc,bi=self.head(h,ctx)
            x=self.feature(h,g,ctx,sc,d);emit=torch.ones(len(ids),dtype=torch.bool) if d==64 else (torch.zeros(len(ids),dtype=torch.bool) if fixed64 or policy is None else policy(x).argmax(-1).eq(1))
            for j,i in enumerate(ids.tolist()):
                traces[i].append({'mode':'EMIT' if bool(emit[j]) else 'THINK','depth':d})
                if emit[j]:
                    a=action(sc[j],bi[j],self.none_index)
                    out[i]=a|{'blue':d<64,'rounds':d,'mode_trace':traces[i],'text':encode_action(a)}
            keep=(~emit).nonzero().flatten()
            if not len(keep):break
            h=h.index_select(0,keep);inputs=inputs.index_select(0,keep);rel=rel.index_select(0,keep)
            ctx=RGBNetwork._slice_context(ctx,keep);ids=ids.index_select(0,keep);prev=d
        if any(o is None for o in out):raise AssertionError('unresolved prediction')
        return out

# Text is a lossless action serialization, NOT an explanation of latent thought.
def encode_action(a):
    if a['stop']:return 'STOP'
    p=a['position'];v=a['value']
    if type(p) is not int or type(v) is not int or not 0<=p<1024 or not 0<=v<256:raise ValueError('action range')
    return f'PATCH {p} {v:02x}'
def decode_action(text):
    if text=='STOP':return {'position':1024,'value':None,'stop':True}
    m=re.fullmatch(r'PATCH (0|[1-9][0-9]{0,3}) ([0-9a-f]{2})',text)
    if not m or int(m[1])>=1024:raise ValueError('invalid action text')
    return {'position':int(m[1]),'value':int(m[2],16),'stop':False}
def project(rows):return [{'command':r['command'],'binary':r['binary']} for r in rows]
def load_actor(seed):
    model=HybridNetwork(seed);d=torch.load(ROOT/f'models/actor-{seed}.pt',map_location='cpu',weights_only=True)
    if d['architecture']!=ARCH:raise ValueError('architecture')
    model.link.load_state_dict(d['link']);model.head.load_state_dict(d['head']);return model.eval()
def load_policy(seed,mode):
    p=ModePolicy();d=torch.load(ROOT/f'models/policy-{mode}-{seed}.pt',map_location='cpu',weights_only=True)
    p.load_state_dict(d['policy']);return p.eval()
