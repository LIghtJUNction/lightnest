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
    def __init__(self, width=96, chunk_size=64, heads=4, landmarks=8, top_k=16):
        super().__init__()
        self.width, self.chunk_size = width, chunk_size
        self.byte = nn.Embedding(257, 8, padding_idx=0)
        self.encode_chunk = nn.Sequential(nn.Linear(chunk_size * 8, width), nn.GELU(), nn.LayerNorm(width))
        self.kind = nn.Embedding(32, width)
        self.position = nn.Linear(8, width, bias=False)
        self.memory = CompressRouteAttention(width, heads, landmarks, top_k)
        self.encoded_chunks = 0
        self._content_cache = OrderedDict()

    def train(self, mode=True):
        self._content_cache.clear()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._content_cache.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def encode(self, observation: Observation, device=None):
        device = device or self.byte.weight.device
        if any(len(p) > self.chunk_size for p in observation.payloads):
            raise ValueError("store/encoder chunk sizes differ")
        raw = torch.zeros(len(observation.payloads), self.chunk_size, dtype=torch.long, device=device)
        for i, payload in enumerate(observation.payloads):
            if payload:
                raw[i, :len(payload)] = torch.tensor(list(payload), device=device) + 1
        cacheable = not self.training and not torch.is_grad_enabled()
        keys = [(payload, str(device), str(self.byte.weight.dtype)) for payload in observation.payloads]
        missing = [i for i, key in enumerate(keys) if not cacheable or key not in self._content_cache]
        encoded = self.encode_chunk(self.byte(raw[missing]).flatten(1)) if missing else None
        features = {}
        for j, i in enumerate(missing):
            features[i] = encoded[j]
            if cacheable:
                self._content_cache[keys[i]] = encoded[j]
        local = torch.stack([features[i] if i in features else self._content_cache[key] for i, key in enumerate(keys)])
        while len(self._content_cache) > 8192:
            self._content_cache.popitem(last=False)
        index = torch.tensor(observation.unique_ids, dtype=torch.long, device=device)
        positions = torch.tensor(observation.addresses, dtype=local.dtype, device=device)
        scales = torch.tensor((1., 64., 4096., 262144.), device=device)
        angle = positions[:, None] / scales
        address = torch.cat((angle.sin(), angle.cos()), -1)
        kinds = torch.tensor(observation.kinds, dtype=torch.long, device=device)
        result = local[index] + self.position(address) + self.kind(kinds)
        self.encoded_chunks += len(missing)
        return result[None], raw

    def build(self, observation, condition):
        features, raw = self.encode(observation)
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        bank = self.memory.build(features, valid, condition)
        return features, raw, bank
