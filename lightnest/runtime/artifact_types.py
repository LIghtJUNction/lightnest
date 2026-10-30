"""Format detection and validation for the Any2Any artifact boundary."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import zipfile

from lightnest.runtime.artifact import inspect_elf
from lightnest.architecture.any2any import KINDS

@dataclass(frozen=True)
class ArtifactInfo:
    kind: str
    size: int
    valid: bool
    metadata: dict


def detect_kind(raw: bytes, filename: str = "") -> str:
    lower = filename.lower()
    if lower.endswith(('.py', '.rs', '.c', '.h', '.cpp', '.md', '.toml', '.yaml', '.yml', '.txt', '.rst', '.sh')):
        return "text"
    if lower.endswith(('.bin', '.dat', '.pyc', '.class', '.wasm', '.woff', '.ttf')):
        return "raw"
    if raw.startswith(b"\x7fELF"): return "elf"
    if raw.startswith(b"MZ"): return "pe"
    if raw.startswith(b"\xcf\xfa\xed\xfe") or raw.startswith(b"\xfe\xed\xfa\xcf"): return "macho"
    if raw.startswith(b"PK\x03\x04"): return "archive"
    if raw.startswith(b"\x1f\x8b") or raw.startswith(b"\xfd7zXZ\x00"): return "compressed"
    if raw.startswith(b"%PDF-"): return "pdf"
    if raw.startswith(b"SQLite format 3\x00"): return "sqlite"
    if lower.endswith(('.json', '.jsonl')):
        try: json.loads(raw); return "json"
        except (UnicodeDecodeError, json.JSONDecodeError): pass
    try:
        text = raw.decode("utf-8")
        if "\x00" not in text and ("," in text or "\n" in text): return "csv" if "," in text else "text"
    except UnicodeDecodeError: pass
    return "raw"


def inspect_artifact(raw: bytes, filename: str = "") -> ArtifactInfo:
    if not raw: raise ValueError("empty artifact")
    kind = detect_kind(raw, filename); metadata = {}
    valid = True
    try:
        if kind == "elf": metadata = inspect_elf(raw)
        elif kind == "json": json.loads(raw)
        elif kind == "archive":
            with zipfile.ZipFile(__import__('io').BytesIO(raw)) as archive: metadata = {"members": archive.namelist()[:4096]}
        elif kind == "sqlite":
            connection = sqlite3.connect(":memory:"); connection.deserialize(raw); connection.execute("pragma integrity_check"); connection.close()
    except Exception as error:
        valid = False; metadata = {"error": type(error).__name__}
    return ArtifactInfo(kind, len(raw), valid, metadata)
