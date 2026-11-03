"""Learned complete-image decoder; output is never combined with an ELF template."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def condition_tensor(specs, device="cpu"):
    result = []
    for spec in specs:
        if set(spec) == {"exit"} and type(spec["exit"]) is int and 0 <= spec["exit"] <= 255:
            mode, values = 0, [spec["exit"], *([0] * 8)]
        elif set(spec) == {"stdout"} and isinstance(spec["stdout"], str):
            payload = bytes.fromhex(spec["stdout"])
            if len(payload) != 8:
                raise ValueError("current learned stdout curriculum uses exactly eight requested bytes")
            mode, values = 1, [0, *payload]
        else:
            raise ValueError("generation request must specify exit status or eight-byte stdout")
        result.append([mode, *values])
    return torch.tensor(result, dtype=torch.long, device=device)


class ELFGenerator(nn.Module):
    """Architecture compiled from wgt/generate.wgt."""


def generation_loss(output, targets, valid):
    eos = targets.eq(256) & valid
    bytes_mask = valid & ~eos
    expected = ((targets[..., None] >> torch.arange(8, device=targets.device)) & 1).float()
    loss = F.binary_cross_entropy_with_logits(output["bits"][bytes_mask], expected[bytes_mask])
    auxiliary = F.binary_cross_entropy_with_logits(output["blue_bits"][bytes_mask], expected[bytes_mask])
    stops = F.binary_cross_entropy_with_logits(output["stop"], eos.float(), reduction="none")
    stop_positive = stops[eos].mean()
    stop_negative = stops[valid & ~eos].mean()
    return loss + .2 * auxiliary + .5 * (stop_positive + stop_negative)


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "generate.wgt", globals()))
