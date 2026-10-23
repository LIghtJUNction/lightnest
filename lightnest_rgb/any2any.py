"""Any2Any artifact model primitives shared by generation and editing.

The artifact is a typed byte stream. Format-specific validators stay at the
boundary; the neural core receives bytes, kind, metadata and observable
feedback rather than an ELF-specific assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn


KINDS = ("raw", "text", "json", "csv", "elf", "pe", "macho", "archive", "compressed", "image", "audio", "pdf", "sqlite", "bytecode")

@dataclass(frozen=True)
class CopyPasteAction:
    source_start: int
    source_end: int
    target_start: int
    operation: str = "copy_replace"

    def __post_init__(self):
        if self.operation not in {"copy_replace","copy_insert"} or min(self.source_start,self.source_end,self.target_start) < 0 or self.source_end <= self.source_start:
            raise ValueError("invalid copy/paste action")


def apply_copy_paste(raw: bytes, action: CopyPasteAction, expected_sha256: str) -> bytes:
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError("artifact identity changed")
    if action.source_end > len(raw) or action.target_start > len(raw): raise ValueError("copy/paste range out of bounds")
    payload=raw[action.source_start:action.source_end]
    if action.operation == "copy_insert": return raw[:action.target_start]+payload+raw[action.target_start:]
    if action.target_start + len(payload) > len(raw): raise ValueError("replacement range out of bounds")
    result=bytearray(raw); result[action.target_start:action.target_start+len(payload)]=payload; return bytes(result)


def kind_index(kind: str) -> int:
    if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
        raise ValueError("artifact kind must be a short nonempty string")
    if kind in KINDS: return KINDS.index(kind)
    return 32 + int.from_bytes(hashlib.sha256(kind.encode()).digest()[:2], "little") % 224


@dataclass(frozen=True)
class ArtifactRequest:
    kind: str
    operation: str
    specification: str = ""

    def __post_init__(self):
        if not isinstance(self.kind, str) or not 1 <= len(self.kind) <= 128 or not self.operation or len(self.operation.encode()) > 512:
            raise ValueError("invalid Any2Any artifact request")


class ArtifactCore(nn.Module):
    """Format-agnostic shared encoder with green observation and red/blue heads."""
    def __init__(self, width=128, heads=4, max_bytes=65536):
        super().__init__()
        if width % heads or width < 32 or max_bytes < 256: raise ValueError("invalid Any2Any geometry")
        self.width, self.max_bytes = width, max_bytes
        self.byte = nn.Embedding(257, width)
        self.kind = nn.Embedding(256, width)
        self.position = nn.Embedding(max_bytes, width)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True), 4)
        self.red = nn.GRUCell(width, width)
        self.abstract = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.LayerNorm(4 * width))
        self.to_blue = nn.Linear(4 * width, width)
        self.kind_head = nn.Linear(width, 256)
        self.summary_head = nn.Linear(width, 5)
        self.copy_query = nn.Linear(width, width, bias=False)
        self.blue_byte = nn.Linear(width, 257)
        self.red_byte = nn.Linear(width, 257)
        self.stop = nn.Linear(width, 1)

    def encode(self, raw: bytes, request: ArtifactRequest):
        if not 1 <= len(raw) <= self.max_bytes: raise ValueError("artifact exceeds Any2Any capacity")
        kind = torch.tensor([kind_index(request.kind)], device=self.byte.weight.device)
        values = torch.tensor(list(raw), device=self.byte.weight.device)[None]
        pos = torch.arange(len(raw), device=values.device)[None]
        return self.encoder(self.byte(values) + self.position(pos) + self.kind(kind)[:, None]), kind

    def encode_chunks(self, raw: bytes, request: ArtifactRequest, chunk_size=256, max_chunks=16):
        if not 32 <= chunk_size <= self.max_bytes or not raw: raise ValueError("invalid hierarchical artifact")
        starts=list(range(0, len(raw), chunk_size))
        if len(starts)>max_chunks:
            starts=[starts[int(i*(len(starts)-1)/(max_chunks-1))] for i in range(max_chunks)]
        chunks=[]
        for start in starts:
            encoded, _ = self.encode(raw[start:start+chunk_size], request)
            chunks.append(encoded.mean(1))
        return torch.stack(chunks, 1)

    @torch.inference_mode()
    def summarize(self, raw: bytes, request: ArtifactRequest, depth=8, chunk_size=256):
        encoded=self.encode_chunks(raw, request, chunk_size)
        abstract,guidance=self.think(encoded, depth)
        return abstract, self.heads(encoded, guidance[:, None])

    def think(self, encoded, depth=8):
        pooled = encoded.mean(1); hidden = pooled
        for _ in range(depth): hidden = self.red(pooled, hidden)
        abstract = self.abstract(hidden)
        return abstract, self.to_blue(abstract)

    def heads(self, encoded, blue_guidance):
        blue = self.blue_byte(encoded + blue_guidance[:, None])
        red = self.red_byte(encoded)
        kind_context = blue_guidance.mean(1) if blue_guidance.ndim == 3 else blue_guidance
        return {"blue_logits": blue, "red_logits": red, "stop_logits": self.stop(encoded).squeeze(-1),
                "kind_logits": self.kind_head(kind_context), "summary": self.summary_head(kind_context),
                "copy_paste": self.copy_paste_scores(encoded, kind_context)}

    def fingerprint(self, request: ArtifactRequest):
        return hashlib.sha256(f"{request.kind}\0{request.operation}\0{request.specification}".encode()).hexdigest()

    def copy_paste_scores(self, encoded, guidance):
        """Native pointer scores for source interval and destination offset."""
        tokens=encoded.shape[1]; query=self.copy_query(encoded + guidance[:,None]); scores=query @ query.transpose(-1,-2) / self.width**.5
        return {"source_start": scores.mean(1), "source_end": scores.mean(1), "target_start": scores.mean(1), "length": scores.mean(1)}

    @torch.inference_mode()
    def propose_copy_paste(self, raw: bytes, request: ArtifactRequest):
        encoded,_=self.encode(raw,request); abstract,guidance=self.think(encoded,8); scores=self.copy_paste_scores(encoded,guidance)
        matrix=scores["source_start"][0]; flat=matrix.flatten(); index=int(flat.argmax()); start=index//matrix.shape[-1]; target=index%matrix.shape[-1]; end=min(len(raw),start+max(1,min(16,len(raw)-start)))
        return CopyPasteAction(start,end,target,"copy_replace"), {"abstract":abstract,"scores":scores}
