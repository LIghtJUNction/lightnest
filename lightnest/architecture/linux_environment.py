"""Learned, bounded approximation of one Linux system call outcome."""
from __future__ import annotations

import torch
from torch import nn


class LearnedOpenat(nn.Module):
    """Architecture compiled from wgt/linux_environment.wgt."""


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "linux_environment.wgt", globals()))
