"""Experimental action space for an explicitly single-bit repair task.

The position and the bit are learned. Only the task's one-bit corruption
contract is fixed; neither clean bytes nor a fault location enter inference.
"""
from __future__ import annotations

import torch

from .delta_policy import DeltaRGBPolicy


class SingleBitRGBPolicy(DeltaRGBPolicy):
    def readout(self, state, *, blue=False):
        output = super().readout(state, blue=blue)
        flip_logits = output['bits'] * state['xor_sign']
        choices = flip_logits.argmax(-1, keepdim=True)
        one_flip = torch.full_like(flip_logits, -1).scatter(-1, choices, 1)
        return output | {'bits': one_flip * state['xor_sign'],
                         'flip_logits': flip_logits, 'edit_encoding': 'single_bit'}
