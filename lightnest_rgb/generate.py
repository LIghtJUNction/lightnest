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
    def __init__(self, width=128, capacity=2048):
        super().__init__()
        self.width, self.capacity = width, capacity
        self.byte = nn.Embedding(257, width)  # 256 is BOS, never an emitted file byte.
        self.position = nn.Embedding(capacity + 1, width)
        self.mode = nn.Embedding(3, width)
        self.condition = nn.Linear(9 * 8, width)
        self.decode = nn.GRU(2 * width, width, 2, batch_first=True)
        self.red = nn.GRUCell(2 * width, width)
        self.norm = nn.LayerNorm(width)
        self.blue_bits = nn.Linear(width, 8)
        self.red_bits = nn.Linear(width, 8)
        self.stop = nn.Linear(width, 1)
        self.copy_query = nn.Linear(width, width, bias=False)
        self.copy_keys = nn.Embedding(9, width)
        self.copy_gate = nn.Linear(width, 1)
        nn.init.constant_(self.copy_gate.bias, -2)

    def context(self, condition):
        values = condition[:, 1:]
        bits = ((values[..., None] >> torch.arange(8, device=values.device)) & 1).float()
        return self.condition(bits.flatten(1)) + self.mode(condition[:, 0]), bits

    def forward(self, previous, condition, *, red_depth=2, hidden=None, position_offset=0):
        if previous.ndim != 2 or not 0 <= red_depth <= 16 or previous.shape[1] + position_offset > self.capacity + 1:
            raise ValueError("generation sequence or depth exceeds capacity")
        context, requested_bits = self.context(condition)
        position = self.position(torch.arange(position_offset, position_offset + previous.shape[1], device=previous.device))
        inputs = self.byte(previous) + position
        repeated = context[:, None].expand_as(inputs)
        base, hidden = self.decode(torch.cat((inputs, repeated), -1), hidden)
        thought = base
        for _ in range(red_depth):
            thought = self.norm(self.red(torch.cat((base, repeated), -1).flatten(0, 1), thought.flatten(0, 1)).reshape_as(thought))
        scores = self.copy_query(thought) @ self.copy_keys.weight.T / self.width**.5
        copied = torch.matmul(scores.softmax(-1), requested_bits)
        gate = self.copy_gate(thought).sigmoid()
        logits = self.red_bits(thought) * (1 - gate) + (copied * 2 - 1) * 12 * gate
        return {"bits": logits, "blue_bits": self.blue_bits(base), "stop": self.stop(thought).squeeze(-1), "hidden": hidden}

    @torch.inference_mode()
    def generate(self, spec, *, red_depth=2, max_bytes=None):
        if self.training:
            raise RuntimeError("eval() required")
        maximum = min(max_bytes or self.capacity, self.capacity)
        condition = condition_tensor([spec], self.byte.weight.device)
        previous = torch.tensor([[256]], device=self.byte.weight.device); hidden = None; result = bytearray()
        for position in range(maximum + 1):
            output = self(previous, condition, red_depth=red_depth, hidden=hidden, position_offset=position)
            hidden = output["hidden"]
            if not torch.isfinite(output["bits"]).all() or not torch.isfinite(output["stop"]).all():
                raise FloatingPointError("nonfinite generation")
            if float(output["stop"][0, -1].sigmoid()) >= .5:
                return {"bytes": bytes(result), "stop": True, "steps": position + 1}
            if position == maximum:
                break
            value = sum(int(v >= 0) << k for k, v in enumerate(output["bits"][0, -1]))
            result.append(value); previous = torch.tensor([[value]], device=previous.device)
        return {"bytes": bytes(result), "stop": False, "steps": maximum + 1}


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
