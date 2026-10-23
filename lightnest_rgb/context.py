"""Learned note storage and context budgeting for long artifact observations."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NeuralNotebook(nn.Module):
    """A differentiable fixed-slot notebook with a learned keep budget.

    Evidence is written by content-addressed soft routing.  At inference the
    budget head chooses how many slots remain active; training keeps routing
    soft so the whole notebook receives task gradients.
    """

    def __init__(self, width=128, slots=32):
        super().__init__()
        if type(width) is not int or width < 8 or type(slots) is not int or not 2 <= slots <= 256:
            raise ValueError("invalid neural notebook geometry")
        self.width, self.slots = width, slots
        self.slot_keys = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.write_gate = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.query = nn.Linear(width, width, bias=False)
        self.budget = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width),
                                    nn.Linear(width, slots + 1))

    def write(self, evidence, notes=None):
        if evidence.ndim != 3 or evidence.shape[-1] != self.width or not evidence.shape[1]:
            raise ValueError("notebook evidence must be [batch, items, width]")
        if notes is None:
            notes = evidence.new_zeros(evidence.shape[0], self.slots, self.width)
        if notes.shape != (evidence.shape[0], self.slots, self.width):
            raise ValueError("notebook state shape mismatch")
        address = torch.softmax(self.key(evidence) @ self.slot_keys.T / math.sqrt(self.width), -1)
        candidate = torch.einsum("bns,bnd->bsd", address, self.value(evidence))
        gate = torch.sigmoid(self.write_gate(torch.cat((notes, candidate), -1)))
        return notes + gate * (candidate - notes), {"address": address, "write_gate": gate}

    def read(self, notes, query, *, hard=False):
        if notes.ndim != 3 or notes.shape[1:] != (self.slots, self.width):
            raise ValueError("notebook state shape mismatch")
        if query.ndim == 1:
            query = query[None]
        if query.ndim != 2 or query.shape != (notes.shape[0], self.width):
            raise ValueError("notebook query shape mismatch")
        scores = self.query(query) @ self.slot_keys.T / math.sqrt(self.width)
        weights = torch.softmax(scores, -1)
        budget_logits = self.budget(torch.cat((query, notes.mean(1)), -1))
        selected = budget_logits.argmax(-1)
        if hard:
            active = []
            for row, count in zip(weights, selected.tolist(), strict=True):
                if count == 0:
                    active.append(torch.full_like(row, 1 / self.slots))
                    continue
                mask = torch.zeros_like(row)
                mask[row.topk(count).indices] = 1
                active.append(row * mask / row.mul(mask).sum().clamp_min(1e-8))
            weights = torch.stack(active)
        else:
            probabilities = torch.softmax(budget_logits, -1)
            choices = [weights.new_full(weights.shape, 1 / self.slots)]
            for count in range(1, self.slots + 1):
                active = []
                for row in weights:
                    mask = torch.zeros_like(row)
                    mask[row.topk(count).indices] = 1
                    active.append(row * mask / row.mul(mask).sum().clamp_min(1e-8))
                choices.append(torch.stack(active))
            weights = torch.stack(choices, 1)
            weights = torch.einsum("bc,bcs->bs", probabilities, weights)
        context = torch.einsum("bs,bsd->bd", weights, notes)
        return context, weights, budget_logits

    def compress(self, evidence, query=None, notes=None, *, hard=False):
        if query is None:
            query = evidence.mean(1)
        notes, write = self.write(evidence, notes)
        context, weights, budget_logits = self.read(notes, query, hard=hard)
        return {"notes": notes, "context": context, "weights": weights,
                "budget_logits": budget_logits, "write": write}


def notebook_budget_loss(output, target):
    """Train the context budget against a task-derived slot target."""
    if output["budget_logits"].ndim != 2 or target.ndim != 1:
        raise ValueError("invalid notebook budget supervision")
    return F.cross_entropy(output["budget_logits"], target)
