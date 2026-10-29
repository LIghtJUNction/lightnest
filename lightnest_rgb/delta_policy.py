"""Experimental per-bit XOR editing; existing checkpoint semantics are unchanged.

The network predicts which bits to flip. Unchanged bits come from the actual
input bytes through an exact logit-sign transform, without reference programs,
known corruption positions or instruction-specific inverse mutation rules.
"""
from __future__ import annotations

import torch

from .model import RGBPolicy


class DeltaRGBPolicy(RGBPolicy):
    def begin(self, image, command, frames=()):
        state = super().begin(image, command, frames)
        values = torch.tensor(list(image), device=self.command.weight.device)
        bits = ((values[:, None] >> torch.arange(8, device=values.device)) & 1).to(self.command.weight.dtype)
        state['xor_sign'] = 1 - 2 * bits
        return state

    def readout(self, state, *, blue=False):
        output = super().readout(state, blue=blue)
        return output | {'bits': output['bits'] * state['xor_sign'], 'edit_encoding': 'xor'}
