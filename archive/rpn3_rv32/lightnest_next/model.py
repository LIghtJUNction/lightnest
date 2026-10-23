"""RPN3: instruction encoding, global context, recurrent relations, edit sets.

All edit locations and replacement bits are learned. There is no opcode-to-patch
table, output-ancestry oracle, early exit, or execution feedback in prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .isa import graph

ARCHITECTURE = "lightnest-rpn3-v1"
SUPPORTED_COMMANDS = frozenset(
    text
    for source, target in (("add", "sub"), ("sub", "add"))
    for text in (f"change {source} to {target}",
                 f"replace {source} with {target} in the returned computation",
                 f"change the output-producing {source} to {target}",
                 f"change every {source} that contributes to the output to {target}",
                 f"do not change {source} to {target}",
                 f"do not replace {source} with {target} in the returned computation")
)


@dataclass(frozen=True)
class Config:
    width: int = 96
    rounds: int = 16
    heads: int = 4
    max_bytes: int = 1024
    max_command: int = 256
    max_edits: int = 8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or self.width % 4 or self.max_bytes % 4:
            raise ValueError("width must divide into heads and four; byte capacity must be aligned")
        if self.max_bytes > 1024 or self.max_command > 256 or self.max_edits > 8 or self.rounds > 64:
            raise ValueError("configuration exceeds the bounded task contract")


def validate_request(row: dict, config: Config) -> bytes:
    if set(row) != {"command", "binary"}:
        raise ValueError("inference accepts command and binary only")
    if not isinstance(row["command"], str) or not 1 <= len(row["command"].encode()) <= config.max_command:
        raise ValueError("invalid command length")
    if row["command"] not in SUPPORTED_COMMANDS:
        raise ValueError("unsupported request; only the 12 documented training commands are supported")
    if not isinstance(row["binary"], str):
        raise ValueError("binary must be hexadecimal text")
    code = bytes.fromhex(row["binary"])
    if code.hex() != row["binary"].lower() or not 4 <= len(code) <= config.max_bytes or len(code) % 4:
        raise ValueError("invalid aligned program length or hex encoding")
    if code[-4:] != bytes.fromhex("73001000"):
        raise ValueError("program must end in EBREAK")
    graph(code)
    return code


@dataclass
class Batch:
    commands: torch.Tensor
    raw: torch.Tensor
    valid: torch.Tensor
    relations: torch.Tensor

    @classmethod
    def encode(cls, rows: list[dict], config: Config, device="cpu") -> Batch:
        if not rows:
            raise ValueError("empty batch")
        codes = [validate_request(row, config) for row in rows]
        n = max(map(len, codes)) // 4
        commands = torch.zeros(len(rows), config.max_command, dtype=torch.long)
        raw = torch.zeros(len(rows), n, 4, dtype=torch.long)
        valid = torch.zeros(len(rows), n, dtype=torch.bool)
        relations = torch.zeros(len(rows), 2, n, n)
        for j, (row, code) in enumerate(zip(rows, codes, strict=True)):
            text = list(row["command"].encode())
            commands[j, :len(text)] = torch.tensor(text) + 1
            length = len(code) // 4
            raw[j, :length] = torch.tensor(list(code)).reshape(length, 4)
            valid[j, :length] = True
            ids = torch.arange(length)
            relations[j, :, ids, ids] = 1
            for user, definition in graph(code):
                relations[j, 0, user, definition] = 1
                relations[j, 1, definition, user] = 1
        relations /= relations.sum(-1, keepdim=True).clamp_min(1)
        return cls(*(value.to(device) for value in (commands, raw, valid, relations)))


class PatchNetwork(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        d = config.width
        self.command_embedding = nn.Embedding(257, d, padding_idx=0)
        self.command_conv = nn.Conv1d(d, d, 5, padding=2)
        self.byte_embedding = nn.Embedding(256, d // 4)
        self.byte_slot = nn.Embedding(4, d // 4)
        self.fields = nn.ModuleList([nn.Embedding(size, d // 4) for size in (128, 8, 128, 32, 32, 32)])
        self.input = nn.Sequential(nn.Linear(d + 6 * (d // 4), d), nn.LayerNorm(d), nn.GELU())
        self.global_before = nn.TransformerEncoderLayer(d, config.heads, 2*d, dropout=0, batch_first=True, norm_first=True)
        self.message = nn.Linear(3*d, d)
        self.update = nn.GRUCell(d, d)
        self.norm = nn.LayerNorm(d)
        self.global_after = nn.TransformerEncoderLayer(d, config.heads, 2*d, dropout=0, batch_first=True, norm_first=True)
        self.readout = nn.Sequential(nn.Linear(3*d + d//4, d), nn.GELU(), nn.LayerNorm(d))
        self.edit = nn.Linear(d, 1)
        self.replacement = nn.Linear(d, 8)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        c, raw, valid, rel = batch.commands, batch.raw, batch.valid, batch.relations
        command_mask = c.ne(0)
        command = F.gelu(self.command_conv(self.command_embedding(c).transpose(1, 2))).transpose(1, 2)
        command = (command * command_mask[..., None]).sum(1) / command_mask.sum(1, keepdim=True)
        word = raw[..., 0] | raw[..., 1] << 8 | raw[..., 2] << 16 | raw[..., 3] << 24
        values = (word & 127, word >> 12 & 7, word >> 25 & 127, word >> 7 & 31, word >> 15 & 31, word >> 20 & 31)
        byte = self.byte_embedding(raw)
        h = self.input(torch.cat([byte.flatten(-2), *(emb(x) for emb, x in zip(self.fields, values, strict=True))], -1))
        position = torch.arange(raw.shape[1], device=raw.device, dtype=h.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.config.width, 2, device=raw.device, dtype=h.dtype) * (-math.log(10000)/self.config.width))
        pos = torch.stack(((position*frequency).sin(), (position*frequency).cos()), -1).flatten(-2)
        inputs = h + command[:, None] + pos
        h = self.global_before(inputs, src_key_padding_mask=~valid).masked_fill(~valid[..., None], 0)
        for _ in range(self.config.rounds):
            message = self.message(torch.cat((inputs, torch.bmm(rel[:, 0], h), torch.bmm(rel[:, 1], h)), -1))
            h = self.norm(self.update(message.flatten(0, 1), h.flatten(0, 1)).reshape_as(h))
            h = h.masked_fill(~valid[..., None], 0)
        h = self.global_after(h, src_key_padding_mask=~valid).masked_fill(~valid[..., None], 0)
        root = h[torch.arange(len(h), device=h.device), valid.sum(1)-1]
        context = torch.cat((h, root[:, None].expand_as(h), command[:, None].expand_as(h)), -1)
        context = context[:, :, None].expand(-1, -1, 4, -1)
        slots = self.byte_slot(torch.arange(4, device=raw.device))
        decoded = self.readout(torch.cat((context, byte + slots), -1)).flatten(1, 2)
        edit = self.edit(decoded).squeeze(-1).masked_fill(~valid.repeat_interleave(4, 1), -torch.inf)
        return edit, self.replacement(decoded)

    @torch.inference_mode()
    def predict(self, rows: list[dict], threshold: float = 0.5) -> list[dict]:
        if self.training:
            raise RuntimeError("call eval() before prediction")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        batch = Batch.encode(rows, self.config, next(self.parameters()).device)
        logits, bits = self(batch)
        valid = batch.valid.repeat_interleave(4, 1)
        if not torch.isfinite(logits[valid]).all() or not torch.isfinite(bits).all():
            raise FloatingPointError("nonfinite model output")
        result = []
        for row, score, values in zip(rows, logits.sigmoid(), bits, strict=True):
            code = bytes.fromhex(row["binary"])
            positions = (score[:len(code)] > threshold).nonzero().flatten().tolist()
            patches = [{"offset":p, "value":sum(int(v > 0) << k for k, v in enumerate(values[p]))} for p in positions]
            status = "candidate"
            if len(patches) > self.config.max_edits:
                status = "budget"
            elif any(code[p["offset"]] == p["value"] for p in patches):
                status = "no_progress"
            final = bytearray(code)
            if status == "candidate":
                for patch in patches:
                    final[patch["offset"]] = patch["value"]
            result.append({"status":status, "patches":patches, "final":bytes(final).hex(),
                           "stop":status == "candidate", "assurance":"unverified_neural_candidate"})
        return result


def supervised_loss(logits, bits, batch: Batch, rows: list[dict]):
    """Train every required edit; normalize positives/negatives within each row."""
    valid = batch.valid.repeat_interleave(4, 1)
    if len(rows) != len(logits) or not rows or logits.shape != valid.shape or bits.shape != (*logits.shape, 8):
        raise ValueError("labels and model outputs must align")
    targets = torch.zeros_like(logits)
    values = torch.zeros_like(bits)
    for j, row in enumerate(rows):
        code = bytes(batch.raw[j, batch.valid[j]].flatten().tolist())
        target = bytes.fromhex(row["target"])
        if len(code) != len(target):
            raise ValueError("target length differs from input")
        differences = [(i, b) for i, (a, b) in enumerate(zip(code, target, strict=True)) if a != b]
        if len(differences) > 8:
            raise ValueError("target exceeds edit budget")
        for p, value in differences:
            targets[j, p] = 1
            values[j, p] = torch.tensor([(value >> k) & 1 for k in range(8)], device=bits.device)
    safe_logits = logits.masked_fill(~valid, 0)
    losses = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
    positive = targets.bool()
    negative = valid & ~positive
    pos = (losses * positive).sum(1) / positive.sum(1).clamp_min(1)
    neg = (losses * negative).sum(1) / negative.sum(1).clamp_min(1)
    # The worst wrong byte gets explicit feedback, including on correct programs.
    hard_negative = losses.masked_fill(~negative, 0).max(1).values
    location = (pos + neg + hard_negative).mean()
    if positive.any():
        replacement = F.binary_cross_entropy_with_logits(bits[positive], values[positive])
    else:
        replacement = bits.sum() * 0
    return location + 2 * replacement


def save_model(path: Path, model: PatchNetwork, metadata: dict):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"architecture":ARCHITECTURE, "config":asdict(model.config),
                "state_dict":model.state_dict(), "metadata":metadata}, path)


def load_model(path: Path) -> PatchNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["architecture"] != ARCHITECTURE:
        raise ValueError("checkpoint architecture mismatch")
    model = PatchNetwork(Config(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()
