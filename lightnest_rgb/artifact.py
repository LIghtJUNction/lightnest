"""Complete ELF inspection and checked, atomic byte-patch transactions."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import io
import os
from pathlib import Path
import shutil
import tempfile

from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError

MAX_IMAGE = 32 * 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BytePatch:
    offset: int
    before: bytes
    after: bytes

    def __post_init__(self):
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("patch offset must be a nonnegative integer")
        if not isinstance(self.before, bytes) or not isinstance(self.after, bytes):
            raise ValueError("patch contents must be bytes")
        if not self.before or len(self.before) != len(self.after) or self.before == self.after:
            raise ValueError("a patch must change a nonempty, fixed-length byte range")

    def json(self):
        return {"offset": self.offset, "before": self.before.hex(), "after": self.after.hex()}


def apply_patches(raw: bytes, patches: list[BytePatch], expected_sha256: str) -> bytes:
    if digest(raw) != expected_sha256:
        raise ValueError("artifact identity changed")
    ordered = sorted(patches, key=lambda p: p.offset)
    previous_end = 0
    for patch in ordered:
        end = patch.offset + len(patch.before)
        if patch.offset < previous_end or end > len(raw):
            raise ValueError("overlapping or out-of-bounds patches")
        if raw[patch.offset:end] != patch.before:
            raise ValueError("patch precondition changed")
        previous_end = end
    result = bytearray(raw)
    for patch in ordered:
        result[patch.offset:patch.offset + len(patch.after)] = patch.after
    return bytes(result)


def inspect_elf(raw: bytes) -> dict:
    if not 64 <= len(raw) <= MAX_IMAGE:
        raise ValueError("ELF image exceeds inspection capacity")
    try:
        elf = ELFFile(io.BytesIO(raw))
        if elf.num_sections() > 4096 or elf.num_segments() > 1024:
            raise ValueError("ELF table count exceeds capacity")
        sections = []
        for index, section in enumerate(elf.iter_sections()):
            offset, size = int(section.header.sh_offset), int(section.header.sh_size)
            if section.header.sh_type != "SHT_NOBITS" and offset + size > len(raw):
                raise ValueError("ELF section extends outside file")
            sections.append({"index": index, "name": section.name, "type": section.header.sh_type,
                             "offset": offset, "size": size, "address": int(section.header.sh_addr),
                             "flags": int(section.header.sh_flags)})
        segments = []
        interpreter = None
        for segment in elf.iter_segments():
            offset, size = int(segment.header.p_offset), int(segment.header.p_filesz)
            if offset + size > len(raw):
                raise ValueError("ELF segment extends outside file")
            segments.append({"type": segment.header.p_type, "offset": offset, "file_size": size,
                             "memory_size": int(segment.header.p_memsz), "address": int(segment.header.p_vaddr),
                             "flags": int(segment.header.p_flags)})
            if segment.header.p_type == "PT_INTERP":
                interpreter = segment.get_interp_name()
        imports, exports, needed = set(), set(), []
        symbols = elf.get_section_by_name(".dynsym")
        if symbols:
            for symbol in symbols.iter_symbols():
                if symbol.name:
                    (imports if symbol.entry.st_shndx == "SHN_UNDEF" else exports).add(symbol.name)
        dynamic = elf.get_section_by_name(".dynamic")
        if dynamic:
            needed = [tag.needed for tag in dynamic.iter_tags() if tag.entry.d_tag == "DT_NEEDED"]
        entry = int(elf.header.e_entry)
        executable_entry = any(s["type"] == "PT_LOAD" and s["flags"] & 1 and
                               s["address"] <= entry < s["address"] + s["memory_size"] for s in segments)
        return {"sha256": digest(raw), "size": len(raw), "bits": elf.elfclass,
                "machine": elf.header.e_machine, "type": elf.header.e_type,
                "entry": entry, "entry_in_executable_segment": executable_entry,
                "little_endian": elf.little_endian, "interpreter": interpreter,
                "sections": sections, "segments": segments, "needed": needed,
                "imports": sorted(imports), "exports": sorted(exports)}
    except (ELFError, IndexError, KeyError, TypeError) as error:
        raise ValueError(f"invalid ELF structure: {error}") from error


def write_image(path: Path, raw: bytes, *, require_elf=True) -> dict:
    """Write exactly the supplied bytes; never add or repair an ELF header."""
    path = Path(path)
    if require_elf:
        info = inspect_elf(raw)
        if info["type"] not in ("ET_EXEC", "ET_DYN") or not info["entry_in_executable_segment"]:
            raise ValueError("generated image is not an executable ELF")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".image-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(raw); file.flush(); os.fsync(file.fileno())
            temporary.chmod(0o500)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": digest(raw), "size": len(raw)}


def edit_file(source: Path, destination: Path, patches: list[BytePatch], expected_sha256: str) -> dict:
    """Copy-on-write where supported, fixed-range edits, no source overwrite."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new path")
    original = source.read_bytes()
    expected = apply_patches(original, patches, expected_sha256)
    inspect_elf(expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".edit-", delete=False) as output:
        temporary = Path(output.name)
        cloned = False
        try:
            with source.open("rb") as input_file:
                try:
                    fcntl.ioctl(output.fileno(), 0x40049409, input_file.fileno())  # Linux FICLONE
                    cloned = True
                except OSError:
                    shutil.copyfileobj(input_file, output)
            for patch in patches:
                output.seek(patch.offset); output.write(patch.after)
            output.flush(); os.fsync(output.fileno())
            if digest(temporary.read_bytes()) != digest(expected):
                raise ValueError("transaction copy does not match expected output")
            temporary.chmod(source.stat().st_mode & 0o777 & ~0o222)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"source_sha256": expected_sha256, "sha256": digest(expected), "size": len(expected),
            "changed_bytes": sum(a != b for a, b in zip(original, expected, strict=True)),
            "patches": [p.json() for p in patches], "copy_on_write": cloned}
