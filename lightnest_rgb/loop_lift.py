"""Trusted x86-64 loop encoder for finite-predicate research.

NOT a neural binary generator. It embeds a predicted 256-entry Boolean table
and one fixed loop skeleton. Input ABI: RDI points to n readable bytes,
RSI holds n; return RAX. Outside that memory contract behavior is unspecified.
"""
from __future__ import annotations
import struct


def emit_scan(table: bytes, mode: str) -> bytes:
    """Emit count / first-match / true-prefix loops with bounds checks.

    first returns n if no match; prefix returns n if every byte matches.
    No libc, calls, writes through the input pointer, or system calls.
    """
    if not isinstance(table, bytes) or len(table) != 256 or any(x > 1 for x in table):
        raise ValueError("table must be exactly 256 Boolean bytes")
    if mode not in ("count", "first", "prefix"):
        raise ValueError("unsupported scan mode")
    code, labels, fixes = bytearray(), {}, []
    def emit(h: str) -> None:
        code.extend(bytes.fromhex(h))
    def relative(op: str, target: str) -> None:
        emit(op)
        fixes.append((len(code), target))
        code.extend(b"\0"*4)
    emit("31c0 31c9")  # result=0; index=0
    relative("4c8d05", "table")  # lea r8,[rip+table]
    labels["loop"] = len(code)
    emit("4839f1")  # cmp rcx,rsi
    relative("0f83", "done")  # jae done, before any input load
    emit("0fb6140f 410fb61410")  # edx = input[index]; edx = table[edx]
    if mode == "count":
        emit("4801d0")  # result += predicate
    else:
        emit("85d2")  # test edx,edx
        relative("0f85" if mode == "first" else "0f84", "done")
    emit("48ffc1")
    relative("e9", "loop")
    labels["done"] = len(code)
    if mode != "count":
        emit("4889c8")  # result=index (n on exhausted input)
    emit("c3")
    labels["table"] = len(code)
    code.extend(table)
    for offset, target in fixes:
        code[offset:offset+4] = struct.pack("<i", labels[target] - offset - 4)
    return bytes(code)


def emit_demo_elf(table: bytes, mode: str, data: bytes) -> bytes:
    """ELF demo with embedded input; writes one little-endian uint64.

    ELF layout and loop machine instructions are supplied by this backend.
    It does not read stdin and is not a generated general-purpose application.
    The returned bytes are not automatically written or executed.
    """
    if not isinstance(data, bytes) or len(data) > 131072:
        raise ValueError("demo data must contain at most 131072 bytes")
    function = emit_scan(table, mode)
    stub = bytearray.fromhex("488d3d00000000be00000000e80000000050")
    stub += bytes.fromhex("b801000000bf010000004889e6ba080000000f05b83c00000031ff0f05")
    stub[8:12] = struct.pack("<I", len(data))
    stub[13:17] = struct.pack("<i", len(stub)-17)
    stub[3:7] = struct.pack("<i", len(stub)+len(function)-7)
    payload = bytes(stub)+function+data
    base, off = 0x400000, 120
    ident = b"\x7fELF"+bytes([2,1,1,0])+bytes(8)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident,2,62,1,base+off,64,0,0,64,56,1,0,0,0)
    segment = struct.pack("<IIQQQQQQ",1,5,0,base,base,off+len(payload),off+len(payload),4096)
    return header+segment+payload
