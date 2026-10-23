"""Deterministic transaction checks outside the trainable model.

Passage is not proof of semantic correctness. No subprocesses, host execution,
answer lookup, or changes to original files. Candidate bytes need external tests.
"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Edit:
    offset: int
    before: int
    after: int


def candidate(base: bytes, expected_base_sha256: str, edits: list[Edit], *, max_edits=8) -> bytes:
    if not isinstance(base,bytes) or not 4<=len(base)<=1024 or len(base)%4:
        raise ValueError('requires a 4..1024-byte RV32 snapshot')
    if type(max_edits) is not int or not 0<=max_edits<=8:
        raise ValueError('edit limit must be 0..8')
    if hashlib.sha256(base).hexdigest()!=expected_base_sha256:
        raise ValueError('stale or wrong base; re-observe before editing')
    if len(edits)>max_edits: raise ValueError('too many edits')
    seen=set()
    for e in edits:
        if not isinstance(e,Edit) or type(e.offset) is not int or not 0<=e.offset<len(base):
            raise ValueError('invalid edit offset')
        if e.offset in seen: raise ValueError('overlapping edits')
        if any(type(v) is not int or not 0<=v<=255 for v in (e.before,e.after)):
            raise ValueError('invalid byte value')
        if base[e.offset]!=e.before: raise ValueError('old-byte precondition failed')
        if e.before==e.after: raise ValueError('no-change edit is not STOP')
        seen.add(e.offset)
    out=bytearray(base)
    for e in edits: out[e.offset]=e.after
    return bytes(out)
