"""MiTA-inspired global compression plus reusable top-k detail experts.

Independent PyTorch implementation of the compress-and-route construction in
Wen et al., arXiv:2602.01219v6 §3.2. No upstream GPU kernels are vendored.
The query reads concatenated shared/detail KV with ONE softmax normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class EvidenceBank:
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    landmarks: torch.Tensor
    summaries: torch.Tensor
    detail_keys: torch.Tensor
    detail_values: torch.Tensor
    detail_valid: torch.Tensor
    detail_indices: torch.Tensor
    construction_pairs: int




class CompressRouteAttention(nn.Module):
    """Architecture compiled from wgt/attention.wgt."""


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "attention.wgt", globals()))
