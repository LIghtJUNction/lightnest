"""Neural proposals with bounded visible-example checking, not a proof oracle."""
from __future__ import annotations
import heapq
import math
import time

import torch

from .model import Batch,validate
from .targets import canonical_arch,decode


class VerifiedRepair:
    def __init__(self,model,sandbox,*,candidates=256,events=16,wall_seconds=3.0):
        if model.training:raise ValueError('eval model required')
        if type(candidates)is not int or not 1<=candidates<=4096 or type(events)is not int or not 1<=events<=20:
            raise ValueError('invalid candidate/event budget')
        if not math.isfinite(wall_seconds) or not 0<wall_seconds<=30:raise ValueError('invalid wall budget')
        self.model,self.sandbox=model,sandbox
        self.candidates,self.events,self.wall_seconds=candidates,events,wall_seconds

    @torch.inference_mode()
    def repair(self,row):
        code,_=validate(row,self.model.config)
        arch=canonical_arch(row['architecture'])
        if arch!='x86_64' or arch not in self.model.config.trained_architectures:
            raise ValueError('verified execution currently supports trained Linux x86-64 only')
        started=time.perf_counter();inputs=[e['inputs'] for e in row['examples']]
        expected=[e['output'] for e in row['examples']];seen=set();attempts=checks=0
        def output(final,status,stop):
            patches=[{'offset':p,'value':b} for p,(a,b) in enumerate(zip(code,final,strict=True)) if a!=b]
            return {'architecture':arch,'status':status,'stop':stop,'patches':patches,'final':final.hex(),
                    'candidate_attempts':attempts,'native_checks':checks,'visible_cases':len(inputs),
                    'wall_seconds':time.perf_counter()-started,'assurance':'visible_examples_only_not_proven'}
        def check(candidate):
            nonlocal attempts,checks
            if candidate in seen:return False
            seen.add(candidate);attempts+=1
            if sum(a!=b for a,b in zip(code,candidate,strict=True))>self.model.config.max_edits:return False
            try:decode(arch,candidate)
            except ValueError:return False
            checks+=1
            remaining=self.wall_seconds-(time.perf_counter()-started)
            if remaining<=0:return False
            result=self.sandbox.run(candidate,inputs,timeout=min(.1,remaining))
            return result['status']=='ok' and result['outputs']==expected
        if check(code):return output(code,'observed_unchanged',True)
        batch=Batch.encode([row],self.model.config,next(self.model.parameters()).device)
        z,bits=self.model(batch);z=z[0,:len(code)];bits=bits[0,:len(code)]
        if not torch.isfinite(z).all() or not torch.isfinite(bits).all():raise FloatingPointError('nonfinite neural proposal')
        replacements=[sum(int(x>=0)<<k for k,x in enumerate(v)) for v in bits]
        base=bytearray(code)
        selected=[]
        for p in (z>0).nonzero().flatten().tolist():
            base[p]=replacements[p];selected.append(p)
        # Events are learned bit/mask uncertainties, never opcode/answer rules.
        choices=[]
        for p in selected:
            if base[p]!=code[p]:choices.append((abs(float(z[p])),p,base[p]^code[p]))
            for k in range(8):choices.append((abs(float(bits[p,k])),p,1<<k))
        near=sorted((abs(float(z[p])),p) for p in range(len(code)) if p not in selected)[:8]
        for cost,p in near:
            change=base[p]^replacements[p]
            if change:choices.append((cost,p,change))
        unique={}
        for cost,p,mask in choices:
            key=(p,mask);unique[key]=min(cost,unique.get(key,math.inf))
        events=sorted((cost,p,mask) for (p,mask),cost in unique.items())[:self.events]
        queue=[(0.0,())]
        while queue and attempts<self.candidates and time.perf_counter()-started<self.wall_seconds:
            cost,indices=heapq.heappop(queue);candidate=bytearray(base)
            for index in indices:
                _,p,mask=events[index];candidate[p]^=mask
            if check(bytes(candidate)):return output(bytes(candidate),'sample_checked_candidate',True)
            for index in range(indices[-1]+1 if indices else 0,len(events)):
                heapq.heappush(queue,(cost+events[index][0],(*indices,index)))
        return output(code,'budget' if queue else 'abstain',False)
