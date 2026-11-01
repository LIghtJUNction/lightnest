"""A tiny deterministic subset of real NVIDIA PTX integer semantics.

These tensor operations model PTX `add.u32` and `mul.lo.u32` lane-wise. They
are not a PTX parser, a GPU ISA emulator, or a CUDA runtime.
"""
from __future__ import annotations

import torch
from torch import nn


class PTXIntegerOps(nn.Module):
    """Real PTX unsigned 32-bit arithmetic on batched int64 tensors."""

    def forward(self, left: torch.Tensor, right: torch.Tensor, *, operation: str):
        if left.dtype != torch.long or right.dtype != torch.long or left.shape != right.shape:
            raise ValueError('operands must be equal-shape int64 tensors')
        if left.device != right.device:
            raise ValueError('operands must be on the same device')
        a, b = left & 0xffffffff, right & 0xffffffff
        if operation == 'add.u32':
            return (a + b) & 0xffffffff
        if operation == 'mul.lo.u32':
            return (a * b) & 0xffffffff
        raise ValueError('unsupported PTX operation')
