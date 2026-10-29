"""CPU-friendly H-Net-inspired dynamic chunking for the RGB ELF generator.

Implements Hwang et al. (arXiv:2507.07955), equations 4--10, using GRUs
rather than the paper's Mamba/Transformer blocks. This is an experimental,
opt-in decoder, not a reproduction of the large H-Net checkpoints.
The legacy generator and its default checkpoint format are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .generate import ELFGenerator


@dataclass
class ChunkState:
    encoder: Tensor
    main: Tensor
    decoder: Tensor
    key: Tensor
    smooth: Tensor
    offset: int


def ema_scan(values: Tensor, probabilities: Tensor, initial: Tensor) -> Tensor:
    """Causal EMA via an associative scan; avoids unstable division by cumprod."""
    a = 1 - probabilities[..., None]
    z = probabilities[..., None] * values
    stride = 1
    while stride < values.shape[1]:
        z = torch.cat((z[:, :stride], z[:, stride:] + a[:, stride:] * z[:, :-stride]), 1)
        a = torch.cat((a[:, :stride], a[:, stride:] * a[:, :-stride]), 1)
        stride *= 2
    return z + a * initial[:, None]


def ratio_loss(probabilities: Tensor, boundaries: Tensor, target_ratio: float = 4., valid: Tensor | None = None) -> Tensor:
    """H-Net eq. 10. The hard fraction F is intentionally nondifferentiable."""
    if target_ratio <= 1:
        raise ValueError("target_ratio must exceed one")
    if valid is None:
        valid = torch.ones_like(boundaries)
    if valid.shape != probabilities.shape or not bool(valid.any()):
        raise ValueError("ratio loss needs a nonempty valid mask")
    fraction = boundaries[valid].float().mean().detach()
    mean_probability = probabilities[valid].mean()
    n = target_ratio
    return n / (n - 1) * ((n - 1) * fraction * mean_probability + (1 - fraction) * (1 - mean_probability))


class HierarchicalGRU(nn.Module):
    """Drop-in (input, hidden) -> (output, hidden) decoder with hard selection.

    `dense`, `fixed`, and `dynamic` have exactly the same parameter tensors.
    Selected encoder states, NOT future-containing chunk means, enter main.
    Padded main work is measured separately from logical boundary counts.
    """
    def __init__(self, input_size: int, width: int, *, mode: str = "dynamic", target_ratio: int = 4):
        super().__init__()
        if mode not in {"dense", "fixed", "dynamic"} or type(target_ratio) is not int or target_ratio < 2:
            raise ValueError("invalid chunk mode or ratio")
        self.width, self.mode, self.target_ratio = width, mode, target_ratio
        self.encoder = nn.GRU(input_size, width, batch_first=True)
        self.encoder_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.main = nn.GRU(width, width, batch_first=True)
        self.main_norm = nn.LayerNorm(width)
        self.skip = nn.Linear(width, width, bias=False)
        self.decoder = nn.GRU(width, width, batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        # Independent q/k starts with usable boundaries in tiny GRU experiments.
        # This differs from the original implementation's identity initialization.
        nn.init.normal_(self.skip.weight, std=.02)
        self.last_probabilities: Tensor | None = None
        self.last_boundaries: Tensor | None = None
        self.last_padded_main_positions = 0

    def forward(self, inputs: Tensor, state: ChunkState | None = None):
        if inputs.ndim != 3 or inputs.shape[1] < 1:
            raise ValueError("inputs must be a nonempty B,L,D tensor")
        batch, length, _ = inputs.shape
        width = self.width
        if state is None:
            zero = inputs.new_zeros(1, batch, width)
            state = ChunkState(zero, zero.clone(), zero.clone(), inputs.new_zeros(batch, width),
                               inputs.new_zeros(batch, width), 0)
        encoded, encoder_state = self.encoder(inputs, state.encoder)
        encoded = self.encoder_norm(encoded)
        keys = F.normalize(self.key(encoded), dim=-1, eps=1e-6)
        queries = F.normalize(self.query(encoded), dim=-1, eps=1e-6)
        previous_keys = torch.cat((state.key[:, None], keys[:, :-1]), 1)
        probability = ((1 - (queries * previous_keys).sum(-1)) / 2).clamp(0., 1.)
        if state.offset == 0:
            probability = torch.cat((torch.ones_like(probability[:, :1]), probability[:, 1:]), 1)
        if self.mode == "dynamic":
            boundary = probability >= .5
        elif self.mode == "fixed":
            boundary = ((torch.arange(length, device=inputs.device) + state.offset) % self.target_ratio == 0)
            boundary = boundary[None].expand(batch, -1)
            probability = boundary.to(inputs.dtype)
        else:
            boundary = torch.ones(batch, length, dtype=torch.bool, device=inputs.device)
            probability = boundary.to(inputs.dtype)
        self.last_probabilities, self.last_boundaries = probability, boundary
        counts = boundary.sum(1)
        count_max = int(counts.max())
        chunk_ids = boundary.long().cumsum(1) - 1
        if count_max:
            rows, positions = boundary.nonzero(as_tuple=True)
            chunk_indices = chunk_ids[rows, positions]
            selected = encoded.new_zeros(batch, count_max, width)
            selected[rows, chunk_indices] = encoded[rows, positions]
            selected_p = probability.new_zeros(batch, count_max)
            selected_p[rows, chunk_indices] = probability[rows, positions]
            main_raw, _ = self.main(selected, state.main)
            # Carry the final REAL state, not the state after batch padding.
            last_indices = (counts - 1).clamp_min(0)
            last_main = main_raw[torch.arange(batch, device=inputs.device), last_indices]
            main_state = torch.where((counts > 0)[:, None], last_main, state.main[0])[None]
            smoothed = ema_scan(self.main_norm(main_raw), selected_p, state.smooth)
            # Index zero is previous-call context (needed before the first boundary).
            history = torch.cat((state.smooth[:, None], smoothed), 1)
            expanded = history.gather(1, (chunk_ids + 1)[..., None].expand(-1, -1, width))
            last_smooth = history[torch.arange(batch, device=inputs.device), counts]
        else:
            main_state = state.main
            last_smooth = state.smooth
            expanded = state.smooth[:, None].expand(-1, length, -1)
        self.last_padded_main_positions = batch * count_max
        if self.mode == "dynamic":
            confidence = torch.where(boundary, probability, 1 - probability)
            expanded = expanded * (1 + confidence - confidence.detach())[..., None]
        combined = self.skip(encoded) + expanded
        decoded, decoder_state = self.decoder(combined, state.decoder)
        new_state = ChunkState(encoder_state, main_state, decoder_state, keys[:, -1], last_smooth,
                               state.offset + length)
        return self.output_norm(decoded), new_state


class HNetELFGenerator(ELFGenerator):
    """Opt-in generator retaining RGB conditioning, red thought, copy and stop heads."""
    def __init__(self, width=128, capacity=2048, *, chunk_mode="dynamic", target_ratio=4):
        super().__init__(width=width, capacity=capacity)
        self.decode = HierarchicalGRU(2 * width, width, mode=chunk_mode, target_ratio=target_ratio)

    def forward(self, previous, condition, *, red_depth=2, hidden=None, position_offset=0):
        if (hidden is None and position_offset != 0) or (hidden is not None and hidden.offset != position_offset):
            raise ValueError("chunk state and absolute position must agree")
        output = super().forward(previous, condition, red_depth=red_depth, hidden=hidden, position_offset=position_offset)
        output.update(boundary_probabilities=self.decode.last_probabilities,
                      boundaries=self.decode.last_boundaries,
                      padded_main_positions=self.decode.last_padded_main_positions)
        return output
