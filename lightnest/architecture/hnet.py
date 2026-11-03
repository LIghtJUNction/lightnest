"""CPU-friendly H-Net-inspired dynamic chunking for the RGB ELF generator.

Implements Hwang et al. (arXiv:2507.07955), equations 4--10, using GRUs
rather than the paper's Mamba/Transformer blocks. This is an experimental,
opt-in decoder, not a reproduction of the large H-Net checkpoints.
The legacy generator and its default checkpoint format are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .generate import ELFGenerator


@dataclass
class ChunkState:
    encoder: Tensor
    main: Tensor
    decoder: Tensor
    key: Tensor
    smooth: Tensor
    offset: int




def ratio_loss(probabilities: Tensor, boundaries: Tensor, target_ratio: float = 4., valid: Tensor | None = None) -> Tensor:
    """H-Net eq. 10. The hard fraction F is intentionally nondifferentiable."""
    if target_ratio <= 1:
        raise ValueError("target_ratio must exceed one")
    if valid is None:
        valid = torch.ones_like(boundaries)
    if valid.shape != probabilities.shape or not bool(valid.any()):
        raise ValueError("ratio loss needs a nonempty valid mask")
    fraction = boundaries[valid].float().mean().detach()
    mean_probability = probabilities[valid].mean()
    n = target_ratio
    return n / (n - 1) * ((n - 1) * fraction * mean_probability + (1 - fraction) * (1 - mean_probability))


class HierarchicalGRU(nn.Module):
    """Architecture compiled from wgt/hnet.wgt."""


class HNetELFGenerator(ELFGenerator):
    """Architecture compiled from wgt/hnet.wgt."""


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "hnet.wgt", globals()))
