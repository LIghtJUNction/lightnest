"""Bridge the observation ledger to exact next-action reuse.

Only exactly equal model inputs, source version, policy, and weights hit. A
summary or normalized template is NEVER a result-cache key. Neural weights are
fingerprinted on every call; this favors clarity over maximum throughput.
The session is single-threaded and research-only, not a permission boundary.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib,json
import torch
from .runtime import common
from .evidence import EvidenceStore


def fingerprint(model):
    h=hashlib.sha256()
    h.update(json.dumps(asdict(model.config),sort_keys=True).encode())
    h.update(json.dumps(asdict(model.policy),sort_keys=True).encode())
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RGBSession:
    def __init__(self,model,*,cache_entries=128,evidence_bytes=8*1024**2):
        if type(cache_entries) is not int or not 1<=cache_entries<=4096:
            raise ValueError('cache_entries must be 1..4096')
        self.model=model.eval();self.cache_entries=cache_entries;self._cache=OrderedDict()
        self.ledger=EvidenceStore(max_bytes=evidence_bytes)

    @torch.inference_mode()
    def propose(self,program: bytes,request: str,*,namespace: str,source: str,revision: str):
        if not isinstance(request,str) or not 1<=len(request.encode())<=self.model.red.config.command_capacity:
            raise ValueError('invalid request length')
        if not isinstance(program,bytes) or not 4<=len(program)<=self.model.red.config.binary_capacity or len(program)%4:
            raise ValueError('invalid program size')
        ref=self.ledger.observe(program,namespace=namespace,source=source,revision=revision)
        key=(namespace,source,revision,request,program,fingerprint(self.model))
        if key in self._cache:
            saved=self._cache.pop(key);self._cache[key]=saved;out=deepcopy(saved)
            out.update(cache_hit=True,observation_sequence=ref.sequence,recurrent_step_examples_this_call=0)
            return out
        encoded=common.encode([{'command':request,'binary':program.hex()}],self.model.red.config)
        result=self.model(*encoded);pos=int(result['scores'][0].argmax())
        out={'base_sha256':ref.digest,'observation_sequence':ref.sequence,
             'action':'STOP' if pos==self.model.none_index else 'EDIT',
             'source_path':'blue' if bool(result['used_blue'][0]) else 'red',
             'cache_hit':False,'deployment_approved':False,
             'recurrent_step_examples_this_call':result['shared_step_examples']+result['red_step_examples']}
        if pos!=self.model.none_index:
            value=sum(int(result['bits'][0,pos,k]>=0)<<k for k in range(8))
            out['edit']={'offset':pos,'before':program[pos],'after':value}
        self._cache[key]=deepcopy(out)
        while len(self._cache)>self.cache_entries:self._cache.popitem(last=False)
        return out
