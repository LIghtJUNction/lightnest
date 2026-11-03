"""Exact content deduplication with occurrence/address provenance retained."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import math

import torch
from torch import nn

from lightnest.runtime.artifact import digest
from lightnest.architecture.attention import CompressRouteAttention


@dataclass(frozen=True)
class Frame:
    kind: str
    payload: bytes
    address: int = 0


@dataclass
class Observation:
    payloads: list[bytes]
    unique_ids: list[int]
    addresses: list[int]
    kinds: list[int]
    image_sha256: str
    total_payload_bytes: int
    encoded_payload_bytes: int

    @property
    def instances(self):
        return len(self.unique_ids)


class ObservationStore:
    def __init__(self, chunk_size=64, max_instances=65536):
        if type(chunk_size) is not int or not 8 <= chunk_size <= 256 or max_instances < 1:
            raise ValueError("invalid observation capacity")
        self.chunk_size, self.max_instances = chunk_size, max_instances

    def collect(self, image: bytes, frames=()):
        # Frames are exact observations, never target binaries or hidden cases.
        all_frames = [Frame("image", image), *frames]
        payloads, ids, addresses, kinds = [], [], [], []
        lookup = {}; total = 0
        for frame in all_frames:
            if not isinstance(frame, Frame) or frame.address < 0:
                raise ValueError("invalid observation frame")
            kind = int.from_bytes(hashlib.sha256(frame.kind.encode()).digest()[:2], "little") % 32
            offsets = range(0, len(frame.payload), self.chunk_size) if frame.payload else (0,)
            for offset in offsets:
                payload = frame.payload[offset:offset + self.chunk_size]
                # Content embeddings have no address context. Each occurrence
                # gets its own address/kind representation after lookup.
                if payload not in lookup:
                    lookup[payload] = len(payloads); payloads.append(payload)
                ids.append(lookup[payload]); addresses.append(frame.address + offset); kinds.append(kind)
                total += len(payload)
                if len(ids) > self.max_instances:
                    raise ValueError("observation instance capacity exceeded; no silent truncation")
        if not ids:
            raise ValueError("empty observation")
        return Observation(payloads, ids, addresses, kinds, digest(image), total, sum(map(len, payloads)))


class GreenObserver(nn.Module):
    """Architecture compiled from wgt/observe.wgt."""


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "observe.wgt", globals()))
