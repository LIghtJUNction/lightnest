"""A bounded neural candidate loop over actual ELF and x86-64 bytes."""
from __future__ import annotations

import torch
from torch import nn

from lightnest.architecture.elf_tensor_loader import ELFTensorLoader
from lightnest.architecture.ptx_internal import PTXIntegerOps
from lightnest.architecture.x86_internal import EXITED, UNSUPPORTED, X86Internal


class X86World(nn.Module):
    """Architecture compiled from wgt/x86_world.wgt."""


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "x86_world.wgt", globals()))
