"""Shared RGB policy: green evidence, shallow blue, recurrent red decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .observe import GreenObserver, ObservationStore

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

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "red_depths" and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be positive")
        if self.width % self.heads or self.width % 8 or not 8 <= self.chunk_size <= 256:
            raise ValueError("invalid RGB geometry")
        if not self.red_depths or tuple(sorted(set(self.red_depths))) != self.red_depths or self.blue_depth >= self.red_depths[0] or self.red_depths[-1] > 64:
            raise ValueError("thinking depths must strictly increase after blue")


class ByteHead(nn.Module):
    def __init__(self, width, hidden, chunk_size):
        super().__init__()
        self.slot = nn.Embedding(chunk_size, 8)
        self.decode = nn.Sequential(nn.Linear(width + 8, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edit = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 8)

    def forward(self, hidden, byte_embeddings, length):
        n, chunk, _ = byte_embeddings.shape
        slots = self.slot(torch.arange(chunk, device=hidden.device))
        h = hidden[0, :n, None].expand(-1, chunk, -1)
        value = self.decode(torch.cat((h, byte_embeddings + slots), -1))
        return self.edit(value).flatten()[:length], self.value(value).reshape(-1, 8)[:length]


class RGBPolicy(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config=RGBConfig()):
        super().__init__()
        self.config = config; d = config.width
        self.green = GreenObserver(d, config.chunk_size, config.heads, config.landmarks, config.detail_k)
        self.store = ObservationStore(config.chunk_size)
        self.command = nn.Embedding(257, d, padding_idx=0)
        self.initial = nn.LayerNorm(d)
        self.message = nn.Linear(3 * d, d)
        self.thought = nn.GRUCell(d, d)
        self.thought_norm = nn.LayerNorm(d)
        self.depth = nn.Embedding(65, d)
        self.blue = ByteHead(d, d // 2, config.chunk_size)
        self.red = ByteHead(d, d, config.chunk_size)
        self.risk = nn.Sequential(nn.Linear(2 * d + 3, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.recurrent_steps = 0

    def begin(self, image, command, frames=()):
        if not isinstance(command, str) or not 1 <= len(command.encode()) <= 384:
            raise ValueError("invalid command")
        observation = self.store.collect(image, frames)
        tokens = torch.tensor(list(command.encode()), device=self.command.weight.device) + 1
        condition = self.command(tokens).mean(0)[None]
        features, raw, bank = self.green.build(observation, condition)
        hidden = self.initial(features + condition[:, None])
        n = (len(image) + self.config.chunk_size - 1) // self.config.chunk_size
        unique = torch.tensor(observation.unique_ids[:n], device=raw.device)
        byte_embeddings = self.green.byte(raw[unique])
        return {"observation": observation, "condition": condition, "features": features,
                "bank": bank, "hidden": hidden, "byte_embeddings": byte_embeddings,
                "length": len(image), "depth": 0}

    def advance(self, state, depth, mode="routed"):
        if type(depth) is not int or not state["depth"] < depth <= self.config.red_depths[-1]:
            raise ValueError("thinking depth must advance within the shared budget")
        for step in range(state["depth"] + 1, depth + 1):
            memory = self.green.memory.read(state["hidden"], state["bank"], mode=mode)
            repeated = state["condition"][:, None].expand_as(state["hidden"])
            message = self.message(torch.cat((state["features"], memory, repeated), -1)) + self.depth.weight[step]
            shape = state["hidden"].shape
            state["hidden"] = self.thought_norm(self.thought(message.flatten(0, 1), state["hidden"].flatten(0, 1)).reshape(shape))
            self.recurrent_steps += 1
        state["depth"] = depth

    def readout(self, state, *, blue=False):
        logits, values = (self.blue if blue else self.red)(state["hidden"], state["byte_embeddings"], state["length"])
        probabilities = logits.sigmoid()
        top = probabilities.topk(min(2, len(probabilities))).values
        statistics = torch.stack((top[0], top[0] - top[-1], probabilities.mean()))[None]
        pooled = state["hidden"].mean(1)
        risk = self.risk(torch.cat((pooled.detach(), state["condition"].detach(), statistics.detach()), -1)).squeeze()
        return {"logits": logits, "bits": values, "risk_logit": risk, "depth": state["depth"],
                "origin": "blue" if blue else "red"}

    def forward(self, image, command, frames=(), *, depth=None, mode="routed"):
        state = self.begin(image, command, frames)
        self.advance(state, self.config.blue_depth, mode)
        blue = self.readout(state, blue=True)
        self.advance(state, depth or self.config.red_depths[-1], mode)
        red = self.readout(state)
        return blue, red, state

    @torch.inference_mode()
    def thoughts(self, image, command, frames=(), *, mode="routed", fixed_depth=None):
        if self.training:
            raise RuntimeError("eval() required")
        state = self.begin(image, command, frames)
        if fixed_depth is None:
            self.advance(state, self.config.blue_depth, mode)
            yield self.readout(state, blue=True), state
            depths = self.config.red_depths
        else:
            depths = (fixed_depth,)
        for depth in depths:
            self.advance(state, depth, mode)
            yield self.readout(state), state


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
            "confidence": float(logits[index].sigmoid()), "risk": float(output["risk_logit"].sigmoid()),
            "origin": output["origin"], "depth": output["depth"], "no_progress": value == image[index]}
