"""Learned note storage and context budgeting for long artifact observations."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NeuralNotebook(nn.Module):
    """Architecture compiled from wgt/context.wgt."""


def notebook_budget_loss(output, target):
    """Train the context budget against a task-derived slot target."""
    if output["budget_logits"].ndim != 2 or target.ndim != 1:
        raise ValueError("invalid notebook budget supervision")
    return F.cross_entropy(output["budget_logits"], target)


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "context.wgt", globals()))
