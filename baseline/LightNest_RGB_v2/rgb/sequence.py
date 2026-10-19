"""Bounded sequence risk and verified drafts, with raw and committed actions separate.

No evaluator, target, task family or execution outcomes enter these routines.
All candidates are bytes in memory, never native programs run on the host.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable
import torch
from torch import nn
from .runtime import common
from study.evaluate import action

MAX_ACTIONS=9
FEATURES=144

def inputs_only(rows):
    out=[]
    for row in rows:
        if set(row)!={'command','binary'}:raise ValueError('inference accepts command and binary only')
        if not isinstance(row['command'],str) or not isinstance(row['binary'],str):raise TypeError('strings required')
        raw=bytes.fromhex(row['binary'])
        if not raw or len(raw)%4 or len(raw)>1024:raise ValueError('invalid binary length')
        out.append(dict(row))
    return out

def new_state(row):
    return {'code':bytes.fromhex(row['binary']),'actions':[],'stop':False,'status':'active'}

def signature(a):
    return (bool(a['stop']),int(a['position']),None if a['stop'] else int(a['value']))

def commit(state,a,*,origin):
    """Consume ONE action in the global eight-edit-plus-STOP budget."""
    if state['status']!='active':raise ValueError('cannot change terminal state')
    if len(state['actions'])>=MAX_ACTIONS:raise ValueError('global action budget exceeded')
    a=dict(a);a.pop('memory',None);a['step']=len(state['actions']);a['origin']=origin
    a['blue']=origin in ('blue','verified_blue');a.setdefault('rounds',64)
    state['actions'].append(a)
    if a['stop']:
        state['stop']=True;state['status']='stop';return
    if a['step']==MAX_ACTIONS-1:state['status']='budget';return
    p=a['position'];v=a['value']
    if type(p) is not int or not 0<=p<len(state['code']):state['status']='padded_pointer';return
    if type(v) is not int or not 0<=v<256:state['status']='invalid_value';return
    a['before']=state['code'][p]
    if a['before']==v:state['status']='no_progress';return
    code=bytearray(state['code']);code[p]=v;state['code']=bytes(code)

def parse_active(state):
    if state['status']!='active':return False
    try:common.graph(state['code'])
    except ValueError as e:state['status']='parse_error';state['error']=str(e);return False
    return True

class Predictor:
    """Live neural calls; never an output or teacher cache."""
    def __init__(self,model,mode):
        self.model=model;self.mode=mode;self.calls=0;self.rows=0
    @torch.inference_mode()
    def __call__(self,rows):
        rows=inputs_only(rows);self.calls+=1;self.rows+=len(rows)
        o=self.model(*common.encode(rows,self.model.red.config),mode=self.mode,evidence=self.mode=='blue_only')
        outs=[]
        for j in range(len(rows)):
            a=action(o['scores'][j],o['bits'][j],self.model.none_index)
            pp=o['scores'][j].softmax(-1).topk(2).values
            a.update(blue=self.mode=='blue_only',risk=float(o['predicted_blue_error'][j]),
                margin=float(pp[0]-pp[1]),rounds=self.model.config.blue_rounds if self.mode=='blue_only' else self.model.config.red_rounds)
            if self.mode=='blue_only':a['memory']=o['evidence_slots'][j].mean(0).tolist()
            outs.append(a)
        return outs

def continue_red(red:Callable,rows,states):
    """Continue from committed prefix. Never resets the episode's edit budget."""
    for _ in range(MAX_ACTIONS):
        active=[j for j,st in enumerate(states) if parse_active(st)]
        if not active:break
        todo=[{'command':rows[j]['command'],'binary':states[j]['code'].hex()} for j in active]
        for j,a in zip(active,red(todo),strict=True):commit(states[j],a,origin='red')
    for st in states:
        if st['status']=='active':st['status']='budget'
    return states

def finish_draft_features(row,draft):
    acts=draft['state']['actions'];code=draft['state']['code'];mems=draft['memories']
    zero=[0.]*64
    first=mems[0] if mems else zero;last=mems[-1] if mems else zero
    cs=[float(a['confidence']) for a in acts];ms=[float(a.get('margin',0)) for a in acts];rs=[float(a.get('risk',1)) for a in acts]
    mean=lambda vs:sum(vs)/len(vs) if vs else 0.
    sd=math.sqrt(mean([(x-mean(cs))**2 for x in cs]))
    syntax=True
    try:common.graph(code)
    except ValueError:syntax=False
    vals=[math.log1p(len(code))/10,len(acts)/9,sum(not a['stop'] for a in acts)/8,float(draft['state']['stop']),float(syntax),
          min(cs,default=0),mean(cs),cs[-1] if cs else 0,max(cs,default=0),sd,min(ms,default=0),mean(ms),
          rs[0] if rs else 1,rs[-1] if rs else 1,max(rs,default=1),
          sum(x!=y for x,y in zip(bytes.fromhex(row['binary']),code))/8]
    feat=first+last+vals
    if len(feat)!=FEATURES or not all(math.isfinite(x) for x in feat):
        feat=[0.]*FEATURES;draft['state']['status']='fast_nonfinite';draft['state']['stop']=False
    draft['features']=feat
    return draft

def drafts(blue:Callable,rows,max_actions=9):
    """Speculative pre-states carry no labels and never alter caller bytes."""
    rows=inputs_only(rows)
    if not 1<=max_actions<=9:raise ValueError('max_actions must be 1..9')
    ds=[{'state':new_state(r),'prestates':[],'memories':[]} for r in rows]
    for _ in range(max_actions):
        active=[j for j,d in enumerate(ds) if parse_active(d['state'])]
        if not active:break
        todo=[{'command':rows[j]['command'],'binary':ds[j]['state']['code'].hex()} for j in active]
        try:outs=blue(todo)
        except FloatingPointError:
            # Auxiliary failure is isolated by row; red inputs stay unchanged.
            outs=[]
            for r in todo:
                try:outs.extend(blue([r]))
                except FloatingPointError:outs.append(None)
        for j,r,a in zip(active,todo,outs,strict=True):
            if a is None:ds[j]['state']['status']='fast_nonfinite';continue
            ds[j]['prestates'].append(r);ds[j]['memories'].append(a.get('memory',[0.]*64))
            commit(ds[j]['state'],a,origin='blue')
    # A prefix is deliberately not marked STOP or counted as a complete draft.
    for r,d in zip(rows,ds,strict=True):finish_draft_features(r,d)
    return ds

def verified_drafts(blue:Callable,red:Callable,rows,*,draft_steps=4,verify_batch=32):
    rows=inputs_only(rows);ds=drafts(blue,rows,draft_steps);flat=[];where=[]
    for j,d in enumerate(ds):
        for t,r in enumerate(d['prestates']):flat.append(r);where.append((j,t))
    checked={}
    for s in range(0,len(flat),verify_batch):
        for loc,a in zip(where[s:s+verify_batch],red(flat[s:s+verify_batch]),strict=True):checked[loc]=a
    states=[new_state(r) for r in rows];mismatches=0;accepted=0;discarded=0
    for j,(d,st) in enumerate(zip(ds,states,strict=True)):
        da=d['state']['actions'];consumed=0
        for t,b in enumerate(da):
            if st['status']!='active':break
            a=checked[(j,t)];same=signature(a)==signature(b)
            commit(st,a,origin='verified_blue' if same else 'red');consumed+=1
            if same:accepted+=1
            else:mismatches+=1;break
        discarded+=len(da)-consumed
    continue_red(red,rows,states)
    return states,ds,{'verified_prefix_actions':accepted,'mismatches':mismatches,'discarded_suffix_actions':discarded,'red_verify_rows':len(flat)}

class SequenceRisk(nn.Module):
    """Risk of an entire blue proposal, not the next action only."""
    def __init__(self,stats_only=False):
        super().__init__();self.stats_only=bool(stats_only)
        self.net=nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,1))
        self.register_buffer('mean',torch.zeros(FEATURES));self.register_buffer('scale',torch.ones(FEATURES))
    def forward(self,x):
        x=(x-self.mean)/self.scale
        if self.stats_only:x=torch.cat((torch.zeros_like(x[:,:128]),x[:,128:]),1)
        return self.net(x).squeeze(-1)
    def normalize_from(self,x):
        self.mean.copy_(x.mean(0));self.scale.copy_(x.std(0).clamp_min(1e-4))

@torch.inference_mode()
def risk_drafts(blue,red,risk,rows,threshold):
    rows=inputs_only(rows);ds=drafts(blue,rows,9)
    pr=risk(torch.tensor([d['features'] for d in ds],dtype=torch.float32)).sigmoid().tolist()
    states=[];accept=[];todo=[];ids=[]
    for i,(r,d,p) in enumerate(zip(rows,ds,pr,strict=True)):
        use=math.isfinite(p) and p<=threshold and d['state']['stop'] and d['state']['status']=='stop'
        accept.append(use)
        if use:states.append(d['state'])
        else:states.append(new_state(r));todo.append(r);ids.append(i)
    if todo:
        fall=[states[i] for i in ids];continue_red(red,todo,fall)
    return states,ds,{'accepted_drafts':sum(accept),'risks':pr,'accept_mask':accept}
