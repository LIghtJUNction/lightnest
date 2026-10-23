"""Optional compiled loop. Same frozen tensors, same operations and finite checks.

This is a backend experiment, not a learned architectural change. Construction
and warmup are counted separately from steady-state timing. No TorchScript
artifact is trusted from an external source; build it from this pinned code.
"""
from __future__ import annotations
import time,types
import torch
from torch import nn
from .network import load_rgb

class LoopCore(nn.Module):
    def __init__(self,red):
        super().__init__()
        self.definition_message=red.definition_message
        self.use_message=red.use_message
        self.message_norm=red.message_norm
        self.update=red.update
    def forward(self,h:torch.Tensor,inputs:torch.Tensor,relation:torch.Tensor,valid:torch.Tensor,steps:int):
        batch,n,d=h.shape
        for _ in range(steps):
            from_definition=torch.bmm(relation[:,0],self.definition_message(h))
            from_use=torch.bmm(relation[:,1],self.use_message(h))
            messages=torch.tanh(self.message_norm(inputs+from_definition+from_use))
            proposed=self.update(messages.reshape(-1,d),h.reshape(-1,d)).reshape(batch,n,d)
            proposed=proposed.masked_fill(~valid[...,None],0.)
            if not bool(torch.isfinite(proposed).all()):
                raise RuntimeError('nonfinite shared-depth graph update')
            h=proposed.masked_fill(~valid[...,None],0.)
        return h

def _loop(self,h,inputs,rel,ctx,steps):
    # The compiled graph keeps tensor references, not separately trained weights.
    # Guard against accidental optimizer use or replacing parent Parameter objects.
    if self.training:raise RuntimeError('compiled adapter is inference-only')
    if tuple(id(p) for p in self.red.parameters())!=self._compiled_parent_ids:
        raise RuntimeError('parent tensors replaced; rebuild compiled adapter')
    try:return self._compiled_loop(h,inputs,rel,ctx.valid,steps)
    except torch.jit.Error as e:
        if 'nonfinite shared-depth graph update' in str(e):raise FloatingPointError('nonfinite compiled graph update') from e
        raise

def _encode(self,command,binary,relations):
    inputs,rel,ctx=self.red.encode_inputs(command,binary,relations)
    h=_loop(self,inputs,inputs,rel,ctx,self.config.blue_rounds)
    return inputs,rel,ctx,h

def _continue(self,h,inputs,relations,context):
    h=_loop(self,h,inputs,relations,context,self.config.red_rounds-self.config.blue_rounds)
    return self.red.decode_state(h,context)

def compile_rgb(model):
    if model.training:raise RuntimeError('eval required')
    start=time.process_time();wall=time.perf_counter()
    engine=torch.jit.script(LoopCore(model.red).eval())
    # This is a derived execution engine, not another neural parameter set.
    object.__setattr__(model,'_compiled_loop',engine)
    object.__setattr__(model,'_compiled_parent_ids',tuple(id(p) for p in model.red.parameters()))
    object.__setattr__(model,'encode_shared',types.MethodType(_encode,model))
    object.__setattr__(model,'_continue_red',types.MethodType(_continue,model))
    object.__setattr__(model,'backend_setup',{'cpu_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall})
    return model

def load_compiled_rgb(path):return compile_rgb(load_rgb(path))
