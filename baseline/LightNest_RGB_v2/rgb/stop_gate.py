"""Second-stage research gate: blue may STOP, but cannot issue EDIT actions.

This intentionally narrows blue's role. It is NOT a solution to fast general
editing. Calibrate on public full sequences; freeze; confirm on a new holdout.
The unchanged RGB v1 implementation and original policies remain archived.
"""
from dataclasses import dataclass
import torch
from .network import RGBNetwork,RoutePolicy,load_rgb

@dataclass(frozen=True)
class StopOnlyPolicy(RoutePolicy):
    stop_only: bool = True
    def __post_init__(self):
        super().__post_init__()
        if self.stop_only is not True or self.max_predicted_error>=1:
            raise ValueError('STOP-only gate needs threshold strictly below1')

class StopOnlyRGB(RGBNetwork):
    def fast_readout(self,context,h):
        evidence,blue,risk=super().fast_readout(context,h)
        proposed_edit=blue[0].argmax(-1).ne(self.none_index)
        return evidence,blue,torch.where(proposed_edit,torch.full_like(risk,30.),risk)
    @torch.inference_mode()
    def forward(self,*args,**kwargs):
        if self.policy.enabled and not isinstance(self.policy,StopOnlyPolicy):
            raise ValueError('enabled StopOnlyRGB requires StopOnlyPolicy')
        result=super().forward(*args,**kwargs)
        if kwargs.get('mode','auto')=='auto':
            chosen=result['scores'].argmax(-1)
            if bool((result['used_blue'] & chosen.ne(self.none_index)).any()):
                raise AssertionError('STOP-only invariant violated')
        result['stop_only_gate']=True
        return result

def load_stop_only(path):
    base=load_rgb(path);m=StopOnlyRGB(base.red,base.config)
    m.load_state_dict(base.state_dict());m.policy=StopOnlyPolicy();return m.eval()
