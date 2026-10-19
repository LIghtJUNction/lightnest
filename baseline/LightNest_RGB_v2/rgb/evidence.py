"""Lossless observation ledger before neural compression.

Deduplicate storage, not event identity, timing, or instruction occurrences.
Byte equality does not imply program equivalence. No neural-state caching is
performed here. Large-window retrieval and learned eviction are future work.
"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ObservationRef:
    sequence: int
    namespace: str
    source: str
    revision: str
    digest: str
    is_repeat: bool


class EvidenceStore:
    def __init__(self, *, max_bytes=8*1024**2, max_events=8192):
        if type(max_bytes) is not int or type(max_events) is not int or min(max_bytes,max_events)<1:
            raise ValueError('positive integer limits required')
        self.max_bytes=max_bytes;self.max_events=max_events
        self._blobs={};self._versions={};self._last={};self._events=[];self.stored_bytes=0

    def observe(self, payload: bytes, *, namespace: str, source: str, revision: str):
        if not isinstance(payload,bytes) or not payload:
            raise ValueError('nonempty raw bytes required')
        if any(not isinstance(s,str) or not s or len(s)>1024 for s in (namespace,source,revision)):
            raise ValueError('bounded namespace, source and revision required')
        digest=hashlib.sha256(payload).hexdigest();key=(namespace,digest)
        version=(namespace,source,revision)
        if version in self._versions and self._versions[version]!=digest:
            raise ValueError('same source revision has conflicting content')
        if key in self._blobs and self._blobs[key]!=payload:
            raise ValueError('content-hash collision')
        added=0 if key in self._blobs else len(payload)
        if self.stored_bytes+added>self.max_bytes or len(self._events)>=self.max_events:
            raise BufferError('budget exhausted; checkpoint/retrieve explicitly, do not silently discard')
        repeat=version in self._versions
        ref=ObservationRef(len(self._events),namespace,source,revision,digest,repeat)
        self._blobs.setdefault(key,payload);self.stored_bytes+=added
        self._versions[version]=digest;self._last[(namespace,source)]=ref;self._events.append(ref)
        return ref

    def read(self, ref: ObservationRef, *, namespace: str):
        if namespace!=ref.namespace: raise PermissionError('namespace mismatch')
        if not 0<=ref.sequence<len(self._events) or self._events[ref.sequence]!=ref:
            raise KeyError('unknown observation reference')
        return self._blobs[(namespace,ref.digest)]

    @property
    def event_count(self): return len(self._events)
