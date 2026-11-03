"""Shared RGB policy: green evidence, shallow blue, recurrent red decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import asyncio

import torch
from torch import nn
from torch.nn import functional as F

from lightnest.architecture.observe import GreenObserver, ObservationStore
from lightnest.architecture.context import NeuralNotebook

ARCHITECTURE = "lightnest-rgb-full-elf-v1"


@dataclass(frozen=True)
class RGBConfig:
    width: int = 64
    chunk_size: int = 64
    heads: int = 4
    landmarks: int = 4
    detail_k: int = 8
    blue_depth: int = 1
    red_depths: tuple[int, ...] = (2, 4, 8, 16)
    note_slots: int = 16

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "red_depths" and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be positive")
        if self.width % self.heads or self.width % 8 or not 8 <= self.chunk_size <= 256 or not 2 <= self.note_slots <= 256:
            raise ValueError("invalid RGB geometry")
        if not self.red_depths or tuple(sorted(set(self.red_depths))) != self.red_depths or self.blue_depth >= self.red_depths[0] or self.red_depths[-1] > 64:
            raise ValueError("thinking depths must strictly increase after blue")


class ByteHead(nn.Module):
    """Architecture compiled from wgt/model.wgt."""


class RGBPolicy(nn.Module):
    """Architecture compiled from wgt/model.wgt."""


def patch_loss(output, image: bytes, target: bytes):
    if len(image) != len(target) or not image:
        raise ValueError("patch supervision needs equal nonempty images")
    logits, bits = output["logits"], output["bits"]
    if logits.shape != (len(image),) or bits.shape != (len(image), 8):
        raise ValueError("patch output does not align with artifact")
    changed = torch.tensor([a != b for a, b in zip(image, target, strict=True)], device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, changed.float(), reduction="none")
    positive = loss[changed].mean() if changed.any() else logits.sum() * 0
    negative = loss[~changed].mean() if (~changed).any() else logits.sum() * 0
    hardest = loss[~changed].max() if (~changed).any() else logits.sum() * 0
    if changed.any():
        targets = torch.tensor([[(target[p] >> k) & 1 for k in range(8)] for p in changed.nonzero().flatten().tolist()],
                               dtype=bits.dtype, device=bits.device)
        replacement = F.binary_cross_entropy_with_logits(bits[changed], targets)
    else:
        replacement = bits.sum() * 0
    return positive + negative + hardest + 2 * replacement


def greedy_action(output, image):
    logits, values = output["logits"], output["bits"]
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise FloatingPointError("nonfinite RGB action")
    index = int(logits.argmax())
    value = sum(int(x >= 0) << k for k, x in enumerate(values[index]))
    return {"stop": bool(logits[index] < 0), "offset": index, "value": value,
            "confidence": float(logits[index].sigmoid().detach()), "risk": float(output["risk_logit"].sigmoid().detach()),
            "origin": output["origin"], "depth": output["depth"], "no_progress": value == image[index]}


def budget_target(outputs, image: bytes, target: bytes):
    """Return the earliest observed budget whose action is task-correct.

    The label is obtained from the target artifact and the model's actual
    action proposals.  It is therefore a utility label from the task, not a
    proxy based on file size, repetition, or a format-specific rule.
    """
    if len(image) != len(target) or not outputs:
        raise ValueError("budget supervision needs equal nonempty images and outputs")
    for index, output in enumerate(outputs):
        action = greedy_action(output, image)
        legal = image == target if action["stop"] else (
            0 <= action["offset"] < len(image)
            and image[action["offset"]] != target[action["offset"]]
            and action["value"] == target[action["offset"]]
        )
        if legal:
            return index
    return len(outputs) - 1


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "model.wgt", globals()))
