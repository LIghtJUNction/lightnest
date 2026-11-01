"""Bounded static ELF64 loader implemented with batched tensor operations."""
from __future__ import annotations

import torch
from torch import nn


class ELFTensorLoader(nn.Module):
    """Load static x86-64 ET_EXEC byte images into a bounded mapping.

    Supports at most eight program headers and 32-bit file offsets / virtual
    addresses. Rejects interpreters, dynamic segments and overlapping mappings.
    This is a conservative ELF subset, not a full Linux loader.
    """

    def __init__(self, memory_size: int = 4096, base: int = 0x400000, max_headers: int = 8):
        super().__init__()
        if not 8 <= memory_size <= 65536 or not 1 <= max_headers <= 8 or base < 0:
            raise ValueError('invalid loader limits')
        self.memory_size, self.base, self.max_headers = memory_size, base, max_headers
        self.register_buffer('elf_magic', torch.tensor([0x7f, 0x45, 0x4c, 0x46], dtype=torch.long), persistent=False)
        self.register_buffer('addresses', torch.arange(memory_size, dtype=torch.long), persistent=False)
        self.register_buffer('header_indices', torch.arange(max_headers, dtype=torch.long), persistent=False)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor):
        if images.ndim != 2 or images.dtype != torch.long or images.shape[1] < 64:
            raise ValueError('images must be int64 with shape [batch, capacity >= 64]')
        if lengths.shape != (images.shape[0],) or lengths.dtype != torch.long:
            raise ValueError('length shape or dtype mismatch')
        if images.device != self.addresses.device or lengths.device != images.device:
            raise ValueError('loader and inputs must be on the same device')
        batch, capacity = images.shape
        valid = (lengths >= 64) & (lengths <= capacity) & ((images >= 0) & (images <= 255)).all(dim=1)

        def read(offset, count):
            positions = offset[..., None] + torch.arange(count, device=images.device)
            inside = (positions >= 0) & (positions < lengths[:, None, None]) & (positions < capacity)
            data = images.gather(1, positions.clamp(0, capacity - 1).reshape(batch, -1)).reshape(*offset.shape, count)
            value = torch.zeros_like(offset)
            for i in range(min(count, 4)):
                value = value | (data[..., i] << (8 * i))
            upper_zero = data[..., 4:].eq(0).all(dim=-1) if count > 4 else torch.ones_like(value, dtype=torch.bool)
            return value, inside.all(dim=-1) & upper_zero

        def scalar(offset, count):
            value, ok = read(torch.full((batch, 1), offset, dtype=torch.long, device=images.device), count)
            return value[:, 0], ok[:, 0]

        magic = images[:, :4].eq(self.elf_magic).all(dim=1)
        typ, ok_type = scalar(16, 2); machine, ok_machine = scalar(18, 2)
        version, ok_version = scalar(20, 4); entry, ok_entry = scalar(24, 8)
        phoff, ok_phoff = scalar(32, 8); ehsize, ok_ehsize = scalar(52, 2)
        entsize, ok_entsize = scalar(54, 2); count, ok_count = scalar(56, 2)
        valid &= (magic & images[:, 4].eq(2) & images[:, 5].eq(1) & images[:, 6].eq(1)
                  & typ.eq(2) & machine.eq(62) & version.eq(1) & ehsize.eq(64) & entsize.eq(56)
                  & (count >= 1) & (count <= self.max_headers) & (phoff + count * 56 <= lengths)
                  & ok_type & ok_machine & ok_version & ok_entry & ok_phoff & ok_ehsize & ok_entsize & ok_count)
        headers = phoff[:, None] + self.header_indices[None] * 56
        ptype, a = read(headers, 4); flags, b = read(headers + 4, 4)
        offset, c = read(headers + 8, 8); address, d = read(headers + 16, 8)
        filesz, e = read(headers + 32, 8); memsz, f = read(headers + 40, 8)
        present = self.header_indices[None] < count[:, None]
        load = present & ptype.eq(1)
        no_dynamic = ~(present & (ptype.eq(2) | ptype.eq(3))).any(dim=1)
        start = address - self.base
        load_ok = (a & b & c & d & e & f & (start >= 0) & (filesz <= memsz)
                   & (start + memsz <= self.memory_size) & (offset + filesz <= lengths[:, None]))
        valid &= no_dynamic & load.any(dim=1) & ((~load) | load_ok).all(dim=1)
        relative = self.addresses[None, None, :] - start[:, :, None]
        mapped = load[:, :, None] & (relative >= 0) & (relative < memsz[:, :, None])
        file_backed = mapped & (relative < filesz[:, :, None])
        file_offsets = (offset[:, :, None] + relative).clamp(0, capacity - 1)
        data = images.gather(1, file_offsets.reshape(batch, -1)).reshape(batch, self.max_headers, self.memory_size)
        valid &= mapped.long().sum(dim=1).le(1).all(dim=1)
        memory = torch.where(file_backed, data, 0).sum(dim=1)
        readable = (mapped & ((flags[:, :, None] & 4) != 0)).any(dim=1)
        executable = (mapped & ((flags[:, :, None] & 1) != 0)).any(dim=1)
        entry_offset = entry - self.base
        entry_inside = (entry_offset >= 0) & (entry_offset < self.memory_size)
        entry_exec = executable.gather(1, entry_offset.clamp(0, self.memory_size - 1)[:, None]).squeeze(1)
        valid &= entry_inside & entry_exec
        return {'memory': memory, 'readable': readable, 'executable': executable,
                'entry': entry_offset, 'valid': valid}
