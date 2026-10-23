"""RV32IM 的一个可执行子集。模型输出指令字节，这里只做解码，不落到宿主机。"""

from __future__ import annotations

import struct
from typing import Literal, NamedTuple

CODE_BASE = 0x100
MEMORY = 4096
EBREAK = 0x00100073

REGS = {
    "zero": 0,
    "ra": 1,
    "sp": 2,
    "t0": 5,
    "t1": 6,
    "t2": 7,
    "a0": 10,
    "a1": 11,
    "a2": 12,
    "t3": 28,
}


def _reg(name: str) -> int:
    if name in REGS:
        return REGS[name]
    if name.startswith("x") and name[1:].isdigit():
        number = int(name[1:])
        if 0 <= number <= 31:
            return number
    raise ValueError(f"未知寄存器 {name}")


def _s32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


def _sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def _imm_i(value: int) -> int:
    if not -2048 <= value <= 2047:
        raise ValueError(f"I 型立即数越界 {value}")
    return value & 0xFFF


def addi(rd: str, rs: str, imm: int) -> int:
    return (_imm_i(imm) << 20) | (_reg(rs) << 15) | (_reg(rd) << 7) | 0x13


def slli(rd: str, rs: str, shamt: int) -> int:
    if not 0 <= shamt <= 31:
        raise ValueError(f"移位量越界 {shamt}")
    return (shamt << 20) | (_reg(rs) << 15) | (1 << 12) | (_reg(rd) << 7) | 0x13


def _shift_right(rd: str, rs: str, shamt: int, arithmetic: bool = False) -> int:
    if not 0 <= shamt <= 31:
        raise ValueError(f"移位量越界 {shamt}")
    return ((0x400 if arithmetic else 0) | shamt) << 20 | _reg(rs) << 15 | 5 << 12 | _reg(rd) << 7 | 0x13


# Standard RV32I register operations; no permissive fallback for reserved encodings.
REGISTER_OPS = {"sll": (0, 1), "slt": (0, 2), "sltu": (0, 3),
                "xor": (0, 4), "srl": (0, 5), "sra": (32, 5),
                "or": (0, 6), "and": (0, 7)}


def _op(funct7: int, funct3: int, rd: str, rs1: str, rs2: str) -> int:
    return (
        (funct7 << 25)
        | (_reg(rs2) << 20)
        | (_reg(rs1) << 15)
        | (funct3 << 12)
        | (_reg(rd) << 7)
        | 0x33
    )


def add(rd: str, rs1: str, rs2: str) -> int:
    return _op(0x00, 0, rd, rs1, rs2)


def sub(rd: str, rs1: str, rs2: str) -> int:
    return _op(0x20, 0, rd, rs1, rs2)


def mul(rd: str, rs1: str, rs2: str) -> int:
    return _op(0x01, 0, rd, rs1, rs2)


def lw(rd: str, imm: int, rs: str) -> int:
    return (_imm_i(imm) << 20) | (_reg(rs) << 15) | (0x2 << 12) | (_reg(rd) << 7) | 0x03


def _branch(funct3: int, rs1: str, rs2: str, imm: int) -> int:
    if imm % 2 != 0 or not -4096 <= imm <= 4094:
        raise ValueError(f"分支偏移无效 {imm}")
    packed = imm & 0x1FFF
    return (
        ((packed >> 12) & 1) << 31
        | ((packed >> 5) & 0x3F) << 25
        | (_reg(rs2) << 20)
        | (_reg(rs1) << 15)
        | (funct3 << 12)
        | ((packed >> 1) & 0xF) << 8
        | ((packed >> 11) & 1) << 7
        | 0x63
    )


def beq(rs1: str, rs2: str, imm: int) -> int:
    return _branch(0, rs1, rs2, imm)


def bne(rs1: str, rs2: str, imm: int) -> int:
    return _branch(1, rs1, rs2, imm)


def blt(rs1: str, rs2: str, imm: int) -> int:
    return _branch(4, rs1, rs2, imm)


def jal(rd: str, imm: int) -> int:
    if imm % 2 != 0 or not -1048576 <= imm <= 1048574:
        raise ValueError(f"跳转偏移无效 {imm}")
    packed = imm & 0x1FFFFF
    return (
        ((packed >> 20) & 1) << 31
        | ((packed >> 1) & 0x3FF) << 21
        | ((packed >> 11) & 1) << 20
        | ((packed >> 12) & 0xFF) << 12
        | (_reg(rd) << 7)
        | 0x6F
    )


def _b_imm(word: int) -> int:
    packed = (
        ((word >> 31) & 1) << 12
        | ((word >> 7) & 1) << 11
        | ((word >> 25) & 0x3F) << 5
        | ((word >> 8) & 0xF) << 1
    )
    return _sext(packed, 13)


def _j_imm(word: int) -> int:
    packed = (
        ((word >> 31) & 1) << 20
        | ((word >> 12) & 0xFF) << 12
        | ((word >> 20) & 1) << 11
        | ((word >> 21) & 0x3FF) << 1
    )
    return _sext(packed, 21)


def _mem_operand(text: str) -> tuple[int, str]:
    if "(" not in text or not text.endswith(")"):
        raise ValueError(text)
    imm_text, reg_text = text[:-1].split("(", 1)
    return int(imm_text or "0"), reg_text


def _encode_line(text: str, pc: int, labels: dict[str, int]) -> int:
    parts = [part.strip() for part in text.replace(",", " ").split()]
    mnemonic = parts[0]
    if mnemonic == "ebreak" and len(parts) == 1:
        return EBREAK
    if mnemonic == "addi" and len(parts) == 4:
        return addi(parts[1], parts[2], int(parts[3]))
    if mnemonic == "slli" and len(parts) == 4:
        return slli(parts[1], parts[2], int(parts[3]))
    if mnemonic == "add" and len(parts) == 4:
        return add(parts[1], parts[2], parts[3])
    if mnemonic == "sub" and len(parts) == 4:
        return sub(parts[1], parts[2], parts[3])
    if mnemonic == "mul" and len(parts) == 4:
        return mul(parts[1], parts[2], parts[3])
    if mnemonic in REGISTER_OPS and len(parts) == 4:
        f7, f3 = REGISTER_OPS[mnemonic]
        return _op(f7, f3, parts[1], parts[2], parts[3])
    if mnemonic in {"srli", "srai"} and len(parts) == 4:
        return _shift_right(parts[1], parts[2], int(parts[3]), mnemonic == "srai")
    if mnemonic == "lw" and len(parts) == 3:
        imm, base = _mem_operand(parts[2])
        return lw(parts[1], imm, base)
    if mnemonic in {"beq", "bne", "blt"} and len(parts) == 4:
        imm = labels[parts[3]] - pc
        if mnemonic == "beq":
            return beq(parts[1], parts[2], imm)
        if mnemonic == "bne":
            return bne(parts[1], parts[2], imm)
        return blt(parts[1], parts[2], imm)
    if mnemonic == "jal" and len(parts) == 3:
        return jal(parts[1], labels[parts[2]] - pc)
    raise ValueError(f"无法编码 {text}")


def assemble(source: str) -> bytes:
    rows: list[tuple[str, str]] = []
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            rows.append(("label", line[:-1].strip()))
        else:
            rows.append(("op", line))
    labels: dict[str, int] = {}
    pc = CODE_BASE
    placed: list[tuple[int, str]] = []
    for kind, text in rows:
        if kind == "label":
            if text in labels:
                raise ValueError(f"重复标签 {text}")
            labels[text] = pc
        else:
            placed.append((pc, text))
            pc += 4
    words = [_encode_line(text, at, labels) for at, text in placed]
    return b"".join(word.to_bytes(4, "little") for word in words)


def binary_words(code: bytes) -> list[str]:
    if len(code) % 4 != 0:
        raise ValueError("指令流不是 4 字节对齐")
    lines: list[str] = []
    for offset in range(0, len(code), 4):
        word = int.from_bytes(code[offset : offset + 4], "little")
        lines.append(f"{word:032b}")
    return lines


class SimulationResult(NamedTuple):
    value: int | None
    status: Literal["halted", "fuel_exhausted", "rejected"]
    steps: int


def simulate_detailed(code: bytes, inputs: list[int], fuel: int = 400) -> SimulationResult:
    """Return the outcome and number of inspected instructions, including ebreak.

    Input rejection before instruction fetch has zero steps. Existing exceptions
    from invalid Python input types are left unchanged.
    """
    if not code or len(code) % 4 != 0 or CODE_BASE + len(code) > MEMORY:
        return SimulationResult(None, "rejected", 0)
    memory = bytearray(MEMORY)
    for index, value in enumerate(inputs):
        if (index + 1) * 4 > CODE_BASE:
            return SimulationResult(None, "rejected", 0)
        struct.pack_into("<i", memory, index * 4, _s32(int(value)))
    memory[CODE_BASE : CODE_BASE + len(code)] = code
    regs = [0] * 32
    regs[11] = 0
    regs[12] = len(inputs)
    pc = CODE_BASE
    steps = 0
    for _ in range(fuel):
        if pc < 0 or pc + 4 > MEMORY or pc % 4 != 0:
            return SimulationResult(None, "rejected", steps)
        word = int.from_bytes(memory[pc : pc + 4], "little")
        steps += 1
        if word == EBREAK:
            return SimulationResult(regs[10], "halted", steps)
        nxt = _step(word, pc, regs, memory)
        if nxt is None:
            return SimulationResult(None, "rejected", steps)
        regs[0] = 0
        pc = nxt
    return SimulationResult(None, "fuel_exhausted", steps)


def simulate(code: bytes, inputs: list[int], fuel: int = 400) -> int | None:
    return simulate_detailed(code, inputs, fuel).value


def _step(word: int, pc: int, regs: list[int], memory: bytearray) -> int | None:
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F
    if opcode == 0x13 and funct3 == 0:
        regs[rd] = _s32(regs[rs1] + _sext(word >> 20, 12))
        return pc + 4
    if opcode == 0x13 and funct3 == 1 and funct7 == 0:
        regs[rd] = _s32((regs[rs1] & 0xFFFFFFFF) << ((word >> 20) & 0x1F))
        return pc + 4
    if opcode == 0x33 and funct3 == 0 and funct7 in {0x00, 0x20, 0x01}:
        left = regs[rs1]
        right = regs[rs2]
        if funct7 == 0x00:
            regs[rd] = _s32(left + right)
        elif funct7 == 0x20:
            regs[rd] = _s32(left - right)
        else:
            regs[rd] = _s32(left * right)
        return pc + 4
    if opcode == 0x13 and funct3 == 5 and funct7 in {0, 32}:
        value = regs[rs1] if funct7 == 32 else regs[rs1] & 0xFFFFFFFF
        regs[rd] = _s32(value >> rs2)
        return pc + 4
    if opcode == 0x33 and (funct7, funct3) in REGISTER_OPS.values():
        left, right = regs[rs1], regs[rs2]
        shift = right & 31
        values = {
            (0, 1): lambda: left << shift,
            (0, 2): lambda: int(left < right),
            (0, 3): lambda: int((left & 0xFFFFFFFF) < (right & 0xFFFFFFFF)),
            (0, 4): lambda: left ^ right,
            (0, 5): lambda: (left & 0xFFFFFFFF) >> shift,
            (32, 5): lambda: left >> shift,
            (0, 6): lambda: left | right,
            (0, 7): lambda: left & right,
        }
        regs[rd] = _s32(values[funct7, funct3]())
        return pc + 4
    if opcode == 0x03 and funct3 == 2:
        addr = regs[rs1] + _sext(word >> 20, 12)
        if addr < 0 or addr + 4 > MEMORY or addr % 4 != 0:
            return None
        regs[rd] = struct.unpack_from("<i", memory, addr)[0]
        return pc + 4
    if opcode == 0x63 and funct3 in {0, 1, 4}:
        left = regs[rs1]
        right = regs[rs2]
        take = (left == right, left != right, left < right)[(0, 1, 4).index(funct3)]
        return pc + (_b_imm(word) if take else 4)
    if opcode == 0x6F:
        regs[rd] = _s32(pc + 4)
        return pc + _j_imm(word)
    return None
