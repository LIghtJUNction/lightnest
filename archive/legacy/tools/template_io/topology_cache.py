"""Cache only static LightNest graph structure, never numeric model states.

Supported R and I arithmetic operations have the same explicit register read /
write roles. Normalize only those validated fields; retain registers, addresses,
branch offsets, branch kinds and return-register selection. Actual model tokens
are untouched. This adapter is tied to the accompanying runtime source hash.
"""
from functools import lru_cache
import struct

def topology_bytes(binary: bytes) -> bytes:
    if not binary or len(binary)%4 or len(binary)>1024:return binary
    out=[]
    for (w,) in struct.iter_unpack('<I',binary):
        op=w&127;f3=(w>>12)&7;f7=w>>25
        if op==0x33:
            if not ((f7==0 and f3<=7) or (f7==32 and f3 in (0,5)) or (f7==1 and f3==0)):
                return binary # exact parser must still reject this input
            w=(w & ((31<<7)|(31<<15)|(31<<20)))|0x33
        elif op==0x13:
            if not (f3==0 or (f3==1 and f7==0) or (f3==5 and f7 in (0,32))):return binary
            w=(w & ((31<<7)|(31<<15)))|0x13
        out.append(w)
    return struct.pack('<'+'I'*len(out),*out)

class TopologyCache:
    def __init__(self,raw_graph,max_entries=1024):
        self.lookup=lru_cache(maxsize=max_entries)(raw_graph)
    def __call__(self,binary: bytes):return self.lookup(topology_bytes(binary))
    def cache_clear(self):self.lookup.cache_clear()
    def cache_info(self):return self.lookup.cache_info()
