"""Lossless boundary for arbitrary user data formats."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import json
from typing import Any


@dataclass(frozen=True)
class AnyData:
    payload: bytes
    format: str = "raw"
    name: str = ""
    metadata: dict[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.payload, bytes) or not self.payload: raise ValueError("data payload must be nonempty bytes")
        if not isinstance(self.format, str) or not self.format: raise ValueError("data format is required")


def pack(value: Any, *, format: str | None = None, name: str = "") -> AnyData:
    """Pack bytes, text, JSON-like values, or base64 transport objects."""
    if isinstance(value, AnyData): return value
    if isinstance(value, bytes): return AnyData(value, format or "raw", name)
    if isinstance(value, str): return AnyData(value.encode(), format or "text", name)
    if isinstance(value, dict) and set(value) == {"base64", "format"}:
        return AnyData(base64.b64decode(value["base64"], validate=True), value["format"], name)
    return AnyData(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), format or "json", name)


def unpack(data: AnyData, *, format: str | None = None):
    target = format or data.format
    if target in {"raw", "bytes", "elf", "pe", "macho", "archive", "compressed", "image", "audio", "pdf", "sqlite", "bytecode"}:
        return data.payload
    if target in {"text", "csv"}: return data.payload.decode("utf-8")
    if target in {"json", "jsonl"}: return json.loads(data.payload)
    return data.payload
