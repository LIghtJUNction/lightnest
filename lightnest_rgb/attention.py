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


def joint_attention(query, shared_keys, shared_values, detail_keys, detail_values, detail_valid):
    """Per-query attention over shared and routed original evidence together."""
    scale = query.shape[-1] ** -0.5
    shared_scores = torch.matmul(query, shared_keys.transpose(-1, -2)) * scale
    detail_scores = (query[..., None, :] * detail_keys).sum(-1) * scale
    scores = torch.cat((shared_scores, detail_scores.masked_fill(~detail_valid, -torch.inf)), -1)
    weights = scores.softmax(-1)
    m = shared_keys.shape[-2]
    output = torch.matmul(weights[..., :m], shared_values)
    output = output + (weights[..., m:, None] * detail_values).sum(-2)
    return output


class CompressRouteAttention(nn.Module):
    def __init__(self, width=96, heads=4, landmarks=8, top_k=16):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (width, heads, landmarks, top_k)) or width % heads:
            raise ValueError("invalid attention dimensions")
        self.width, self.heads, self.landmark_count, self.top_k = width, heads, landmarks, top_k
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.condition = nn.Linear(width, width, bias=False)
        self.landmark_bias = nn.Parameter(torch.randn(heads, landmarks, width // heads) / math.sqrt(width))
        self.output = nn.Linear(width, width, bias=False)
        self.bank_builds = 0
        self.reads = 0

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.width // self.heads).transpose(1, 2)

    def build(self, observations, valid, condition):
        if observations.ndim != 3 or valid.shape != observations.shape[:2] or condition.shape != (len(observations), self.width):
            raise ValueError("evidence dimensions do not align")
        if not valid.any(-1).all():
            raise ValueError("each observation needs at least one valid token")
        batch, n, _ = observations.shape
        keys, values = self._heads(self.key(observations)), self._heads(self.value(observations))
        # Pooling gives landmarks input dependence; learned offsets permit an
        # expert to leave its initial region. Empty windows still probe all KV.
        boundaries = torch.linspace(0, n, self.landmark_count + 1, device=observations.device).long()
        pools = []
        for i in range(self.landmark_count):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            mask = valid[:, start:end]
            pooled = (observations[:, start:end] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            pools.append(pooled)
        landmarks = self._heads(self.query(torch.stack(pools, 1))) + self.landmark_bias[None]
        landmarks = landmarks + self._heads(self.condition(condition[:, None]))
        logits = torch.matmul(landmarks, keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1])
        logits = logits.masked_fill(~valid[:, None, None], -torch.inf)
        summaries = torch.matmul(logits.softmax(-1), values)
        indices = logits.topk(min(self.top_k, n), -1).indices
        expanded = indices[..., None].expand(*indices.shape, keys.shape[-1])
        detail_keys = torch.gather(keys[:, :, None].expand(-1, -1, self.landmark_count, -1, -1), 3, expanded)
        detail_values = torch.gather(values[:, :, None].expand(-1, -1, self.landmark_count, -1, -1), 3, expanded)
        detail_valid = torch.gather(valid[:, None, None].expand(-1, self.heads, self.landmark_count, -1), 3, indices)
        self.bank_builds += 1
        return EvidenceBank(keys, values, valid, landmarks, summaries, detail_keys, detail_values,
                            detail_valid, indices, batch * self.heads * self.landmark_count * n)

    def read(self, queries, bank: EvidenceBank, *, mode="routed"):
        if mode not in ("routed", "compressed", "dense"):
            raise ValueError("attention mode must be routed, compressed or dense")
        q = self._heads(self.query(queries))
        if mode == "dense":
            logits = torch.matmul(q, bank.keys.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            result = torch.matmul(logits.masked_fill(~bank.valid[:, None, None], -torch.inf).softmax(-1), bank.values)
        elif mode == "compressed":
            logits = torch.matmul(q, bank.landmarks.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            result = torch.matmul(logits.softmax(-1), bank.summaries)
        else:
            routing = torch.matmul(q, bank.landmarks.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            expert = routing.argmax(-1)
            index = expert[..., None, None].expand(*expert.shape, bank.detail_keys.shape[-2], q.shape[-1])
            keys = torch.gather(bank.detail_keys, 2, index)
            values = torch.gather(bank.detail_values, 2, index)
            valid = torch.gather(bank.detail_valid, 2, expert[..., None].expand(*expert.shape, keys.shape[-2]))
            result = joint_attention(q, bank.landmarks, bank.summaries, keys, values, valid)
        self.reads += 1
        return self.output(result.transpose(1, 2).reshape(queries.shape))
