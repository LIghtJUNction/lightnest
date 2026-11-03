"""Any2Any artifact model primitives shared by generation and editing.

The artifact is a typed byte stream. Format-specific validators stay at the
boundary; the neural core receives bytes, kind, metadata and observable
feedback rather than an ELF-specific assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from lightnest.architecture.context import NeuralNotebook
from lightnest.architecture.byte_tokenizer import ByteBPE

KINDS = ("raw", "text", "json", "csv", "elf", "pe", "macho", "archive", "compressed", "image", "audio", "pdf", "sqlite", "bytecode")
DEFAULT_RED_DEPTHS = (2, 4, 8, 16)
DEFAULT_MAX_BYTES = 1 << 20
DEFAULT_SEQUENCE_LIMIT = 4096

@dataclass(frozen=True)
class CopyPasteAction:
    source_start: int
    source_end: int
    target_start: int
    operation: str = "copy_replace"

    def __post_init__(self):
        if self.operation not in {"copy_replace","copy_insert"} or min(self.source_start,self.source_end,self.target_start) < 0 or self.source_end <= self.source_start:
            raise ValueError("invalid copy/paste action")


def apply_copy_paste(raw: bytes, action: CopyPasteAction, expected_sha256: str) -> bytes:
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError("artifact identity changed")
    if action.source_end > len(raw) or action.target_start > len(raw): raise ValueError("copy/paste range out of bounds")
    payload=raw[action.source_start:action.source_end]
    if action.operation == "copy_insert": return raw[:action.target_start]+payload+raw[action.target_start:]
    if action.target_start + len(payload) > len(raw): raise ValueError("replacement range out of bounds")
    result=bytearray(raw); result[action.target_start:action.target_start+len(payload)]=payload; return bytes(result)


def kind_index(kind: str) -> int:
    if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
        raise ValueError("artifact kind must be a short nonempty string")
    if kind in KINDS: return KINDS.index(kind)
    return 32 + int.from_bytes(hashlib.sha256(kind.encode()).digest()[:2], "little") % 224


@dataclass(frozen=True)
class ArtifactRequest:
    kind: str
    operation: str
    specification: str = ""

    def __post_init__(self):
        if (not isinstance(self.kind, str) or not 1 <= len(self.kind) <= 128 or not self.operation
                or len(self.operation.encode()) > 512 or not isinstance(self.specification, str)
                or len(self.specification.encode()) > 2048):
            raise ValueError("invalid Any2Any artifact request")


class ArtifactCore(nn.Module):
    """Architecture compiled from wgt/any2any.wgt."""


def semantic_similarity(left, right):
    """Learned token matching; padding never participates in either direction."""
    a, b = F.normalize(left['features'], dim=-1), F.normalize(right['features'], dim=-1)
    scores = torch.einsum('bnd,cmd->bcnm', a, b)
    valid = left['valid'][:, None, :, None] & right['valid'][None, :, None, :]
    scores = scores.masked_fill(~valid, -2.)
    forward = (scores.max(-1).values * left['weights'][:, None]).sum(-1)
    backward = (scores.max(-2).values * right['weights'][None]).sum(-1)
    return (forward + backward) / 2


def artifact_budget_target(candidates, target, kind_target=None, summary_target=None):
    """Choose a Pareto-style compute point from observed task losses.

    Each available task head contributes a rank, so byte scale cannot drown
    out kind or structure supervision.  The budget head learns this choice;
    this helper only constructs training labels from task outcomes.
    """
    if not candidates or target.ndim != 2 or target.shape[0] != 1:
        raise ValueError("invalid Any2Any budget supervision")
    metrics = [[] for _ in range(3)]
    for candidate in candidates:
        metrics[0].append(F.cross_entropy(candidate["red_logits"].reshape(-1, 257), target.reshape(-1)).detach().item())
        metrics[1].append(F.cross_entropy(candidate["kind_logits"], kind_target).detach().item() if kind_target is not None else 0.)
        metrics[2].append(F.mse_loss(candidate["summary"], summary_target).detach().item() if summary_target is not None else 0.)
    ranks = [[0] * len(candidates) for _ in metrics]
    for values, rank in zip(metrics, ranks, strict=True):
        for position, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
            rank[index] = position
    scores = [sum(rank[index] for rank in ranks) for index in range(len(candidates))]
    return min(range(len(scores)), key=lambda index: (scores[index], index))


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "any2any.wgt", globals()))
