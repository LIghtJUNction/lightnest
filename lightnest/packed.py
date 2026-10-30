"""Bounded, read-only access to code inside PyInstaller and LightNest envelopes.

This is a format reader, not a semantic model. It never executes a module or
turns container metadata into a description of program behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import marshal
import os
from pathlib import Path
import struct
import sys
import zlib

WRAPPER_MAGIC = b'LIGHTNEST_WRAPPER_V1'
COOKIE_MAGIC = b'MEI\014\013\012\013\016'
COOKIE = struct.Struct('!8sIIII64s')
ENTRY = struct.Struct('!IIIIBc')


@dataclass(frozen=True)
class Member:
    name: str
    offset: int
    size: int
    expanded_size: int
    compressed: bool
    kind: str


class PackedArtifact:
    def __init__(self, path, *, max_member_bytes=1 << 20):
        self.path = Path(path)
        self.file = self.path.open('rb')
        identity = os.fstat(self.file.fileno())
        self.total = identity.st_size
        self.identity = {'device': identity.st_dev, 'inode': identity.st_ino, 'mtime_ns': identity.st_mtime_ns}
        self.bytes_read = 0
        self.limit = max_member_bytes
        self.wrapper = None
        try:
            start, end = 0, self.total
            footer_size = len(WRAPPER_MAGIC) + 16
            if self.total >= footer_size:
                footer = self._read(self.total - footer_size, footer_size)
                if footer.startswith(WRAPPER_MAGIC):
                    base_size, model_size = struct.unpack('<QQ', footer[len(WRAPPER_MAGIC):])
                    start = self.total - footer_size - base_size - model_size
                    end = start + base_size
                    if start <= 0 or base_size < COOKIE.size or model_size < 4:
                        raise ValueError('invalid wrapper spans')
                    if self._read(start, 4) != b'\x7fELF' or self._read(end, 4) != b'GGUF':
                        raise ValueError('wrapper payload identities do not match')
                    self.wrapper = {'base_offset': start, 'base_bytes': base_size,
                                    'language_offset': end, 'language_bytes': model_size}
            self.region = (start, end)
            tail_start = max(start, end - 65536)
            tail = self._read(tail_start, end - tail_start)
            found = tail.rfind(COOKIE_MAGIC)
            if found < 0 or found + COOKIE.size > len(tail):
                raise ValueError('bounded region has no PyInstaller cookie')
            cookie_at = tail_start + found
            _, archive_size, toc_offset, toc_size, self.python_version, _ = COOKIE.unpack_from(tail, found)
            self.archive_start = cookie_at + COOKIE.size - archive_size
            if (self.archive_start < start or toc_size > self.limit
                    or self.archive_start + toc_offset + toc_size > cookie_at):
                raise ValueError('invalid archive table span')
            toc = self._read(self.archive_start + toc_offset, toc_size)
            self.members = {}; cursor = 0
            while cursor < len(toc):
                if len(toc) - cursor < ENTRY.size:
                    raise ValueError('truncated archive table')
                length, offset, size, expanded, compressed, kind = ENTRY.unpack_from(toc, cursor)
                if length <= ENTRY.size or cursor + length > len(toc) or compressed not in (0, 1):
                    raise ValueError('invalid archive entry')
                name = toc[cursor + ENTRY.size:cursor + length].split(b'\0', 1)[0].decode('utf-8')
                absolute = self.archive_start + offset
                if not name or name in self.members or absolute + size > cookie_at:
                    raise ValueError('archive member leaves payload or repeats a name')
                self.members[name] = Member(name, absolute, size, expanded, bool(compressed), kind.decode('ascii'))
                cursor += length
        except Exception:
            self.file.close(); raise

    def _read(self, offset, size):
        if offset < 0 or size < 0 or offset + size > self.total:
            raise ValueError('read leaves the artifact')
        self.file.seek(offset); raw = self.file.read(size); self.bytes_read += len(raw)
        if len(raw) != size:
            raise ValueError('artifact was truncated during inspection')
        return raw

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _inflate(self, payload, expected=None):
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, self.limit + 1)
        if len(raw) > self.limit or not decoder.eof or decoder.unused_data:
            raise ValueError('compressed member exceeds capacity or is incomplete')
        if expected is not None and len(raw) != expected:
            raise ValueError('expanded member size changed')
        return raw

    def read_member(self, name):
        item = self.members[name]
        if item.expanded_size > self.limit or item.size > self.limit + 65536:
            raise ValueError('archive member exceeds reader capacity')
        raw = self._read(item.offset, item.size)
        return self._inflate(raw, item.expanded_size) if item.compressed else raw

    def read_code_member(self, name):
        if self.members[name].kind not in ('s', 'm', 'M'):
            raise ValueError('member is not a Python code entry')
        if self.python_version != sys.version_info.major * 100 + sys.version_info.minor:
            raise ValueError('code entry bytecode version differs from the reader')
        return importlib.util.MAGIC_NUMBER + bytes(12) + self.read_member(name)

    def modules(self):
        modules = []
        for container in self.members.values():
            if container.kind != 'z':
                continue
            if container.compressed or container.size < 12:
                raise ValueError('unsupported compressed or truncated PYZ envelope')
            header = self._read(container.offset, 12)
            if header[:4] != b'PYZ\0' or header[4:8] != importlib.util.MAGIC_NUMBER:
                raise ValueError('PYZ bytecode version differs from the reader')
            toc_offset = struct.unpack('!i', header[8:])[0]
            if not 12 <= toc_offset <= container.size or container.size - toc_offset > self.limit:
                raise ValueError('PYZ table leaves capacity')
            table = marshal.loads(self._read(container.offset + toc_offset, container.size - toc_offset))
            entries = table.items() if isinstance(table, dict) else table
            for name, value in entries:
                kind, offset, size = value
                if (not isinstance(name, str) or min(offset, size) < 0 or offset + size > toc_offset):
                    raise ValueError('invalid PYZ module span')
                modules.append({'container': container.name, 'name': name, 'kind': kind,
                                'offset': container.offset + offset, 'size': size})
        return modules

    def read_module(self, module):
        known = next((row for row in self.modules() if row['name'] == module), None)
        if known is None:
            raise KeyError(module)
        if known['kind'] not in (0, 1) or known['size'] > self.limit + 65536:
            raise ValueError('module has no code or exceeds capacity')
        raw = self._inflate(self._read(known['offset'], known['size']))
        return importlib.util.MAGIC_NUMBER + bytes(12) + raw

    def metadata(self):
        modules = self.modules()
        return {'path': str(self.path), 'file_bytes': self.total, 'wrapper': self.wrapper,
                'file_identity': self.identity,
                'python_version': self.python_version, 'bytes_read': self.bytes_read,
                'members': [asdict(item) for item in self.members.values()], 'modules': modules,
                'scope': 'container observations only; no semantic inference or module execution'}
