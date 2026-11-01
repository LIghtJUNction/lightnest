"""Learned, bounded approximation of one Linux system call outcome."""
from __future__ import annotations

import torch
from torch import nn


class LearnedOpenat(nn.Module):
    """Predict fd 3, ENOENT, or EACCES from a bounded path and open flags.

    The network contains fitted observations, not a filesystem image. It does
    not implement file descriptors, reads, writes, directory traversal, or the
    full Linux permission model.
    """

    def __init__(self, path_size: int = 64, width: int = 16):
        super().__init__()
        if not 1 <= path_size <= 256 or not 1 <= width <= 128:
            raise ValueError('invalid environment dimensions')
        self.path_size = path_size
        self.byte = nn.Embedding(256, width)
        self.net = nn.Sequential(nn.Linear(path_size * width + 2, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, path: torch.Tensor, flags: torch.Tensor):
        if path.ndim != 2 or path.shape[1] != self.path_size or flags.shape != path.shape[:1]:
            raise ValueError('path or flags shape mismatch')
        before_null = torch.cat((torch.ones_like(path[:, :1]), path.ne(0).long().cumprod(dim=1)[:, :-1]), dim=1)
        embeddings = self.byte(path.clamp(0,255)) * before_null[:, :, None]
        modes = torch.stack(((flags & 1).float(), ((flags >> 1) & 1).float()), dim=1)
        return self.net(torch.cat((embeddings.flatten(1), modes), dim=1))
