"""Load bounded static x86-64 ELF segments for the tensor-resident machine."""
from __future__ import annotations

import torch

from lightnest.runtime.artifact import digest, inspect_elf


def load_static_elf(raw: bytes, *, device='cpu', memory_size=4096, base=0x400000):
    info = inspect_elf(raw)
    if info['machine'] != 'EM_X86_64' or info['type'] != 'ET_EXEC' or info['interpreter']:
        raise ValueError('internal machine requires a static x86-64 ET_EXEC ELF')
    if not info['entry_in_executable_segment']:
        raise ValueError('entry point is not executable')
    memory = bytearray(memory_size)
    occupied = bytearray(memory_size)
    readable = bytearray(memory_size)
    executable = bytearray(memory_size)
    segments = [segment for segment in info['segments'] if segment['type'] == 'PT_LOAD']
    if not segments:
        raise ValueError('ELF has no loadable segment')
    for segment in segments:
        start = segment['address'] - base
        size = segment['memory_size']
        source, file_size = segment['offset'], segment['file_size']
        if start < 0 or size < file_size or start + size > memory_size:
            raise ValueError('load segment is outside the bounded mapping')
        contents = raw[source:source + file_size] + bytes(size - file_size)
        for i, byte in enumerate(contents, start):
            if occupied[i] and memory[i] != byte:
                raise ValueError('overlapping load segments disagree')
            occupied[i] = 1; memory[i] = byte
            readable[i] |= bool(segment['flags'] & 4)
            executable[i] |= bool(segment['flags'] & 1)
    entry = info['entry'] - base
    if not 0 <= entry < memory_size or not executable[entry]:
        raise ValueError('entry is outside the bounded executable mapping')
    return {'memory': torch.tensor([list(memory)], dtype=torch.long, device=device),
            'readable': torch.tensor([list(readable)], dtype=torch.bool, device=device),
            'executable': torch.tensor([list(executable)], dtype=torch.bool, device=device),
            'entry': torch.tensor([entry], dtype=torch.long, device=device),
            'image_sha256': digest(raw), 'base': base}
