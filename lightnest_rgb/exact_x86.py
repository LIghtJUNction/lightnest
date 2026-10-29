"""Bounded x86-64 ELF research interpreter with exact tensor data operations.

The host loads ELF, decodes with iced-x86, and dispatches instructions. Registers
and data memory are bits; arithmetic uses FixedWordAdder. This is neither learned
RGB execution nor a CPU compiled entirely into model weights. No native process
or reference implementation supplies intermediate machine state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json

import iced_x86 as x86
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError
import torch
from torch.nn import functional as F

from .artifact import digest
from .exact_core import ExactBitRAM, FixedWordAdder, decode_words, encode_words
from .execute import Case

SPEC = "lightnest-exact-x86-p1-v1"
MASK64 = (1 << 64) - 1
STACK = 0x7FFF_FFFF_E000
IMMEDIATES = {getattr(x86.OpKind, name) for name in dir(x86.OpKind) if name.startswith('IMMEDIATE')}
MNEMONICS = {value: name.lower() for name, value in vars(x86.Mnemonic).items() if name.isupper()}


class MachineStop(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


@dataclass
class Region:
    start: int
    flags: int
    ram: ExactBitRAM
    state: torch.Tensor
    known: list[bool]
    code: bytes | None = None

    @property
    def end(self):
        return self.start + self.state.shape[1]


class ExactX86Runner:
    """Fresh state per run; static ET_EXEC only, no libc, argv or native fallback.

    Supported system calls are stdin read, stdout/stderr write and exit. Mapped
    regions and the stack are at most 4096 bytes each, total at most 16384 bytes.
    Undefined input state, unsupported semantics and budget exhaustion are typed
    failures, never successful executions. Timings are diagnostic, not a speed
    claim about the tensor core.
    """
    def __init__(self, *, dtype=torch.float32, max_steps=4096, trace_limit=128):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('dtype must be float32 or bfloat16')
        if type(max_steps) is not int or not 1 <= max_steps <= 100000:
            raise ValueError('max_steps must be in 1..100000')
        if type(trace_limit) is not int or not 0 <= trace_limit <= 4096:
            raise ValueError('trace_limit must be in 0..4096')
        self.dtype, self.max_steps, self.trace_limit = dtype, max_steps, trace_limit
        self.adders = {width: FixedWordAdder(width).to(dtype=dtype) for width in (8, 16, 32)}
        self.register_ram = ExactBitRAM(32, 32)

    def bits(self, value, width):
        value &= (1 << width) - 1
        if width == 64:
            return torch.cat((self.bits(value & 0xFFFF_FFFF, 32), self.bits(value >> 32, 32)))
        return encode_words(torch.tensor([value]), width, dtype=self.dtype)[0]

    @staticmethod
    def integer(bits):
        if len(bits) == 64:
            halves = decode_words(bits.reshape(2, 32)).tolist()
            return halves[0] | (halves[1] << 32)
        return int(decode_words(bits[None])[0])

    def ram_access(self, region, row, value=None):
        key = F.one_hot(torch.tensor([row]), region.ram.rows).to(self.dtype)
        region.state, old = region.ram(region.state, key,
                                      torch.zeros(1, region.ram.word_bits, dtype=self.dtype) if value is None else value[None],
                                      torch.tensor([[int(value is not None)]], dtype=self.dtype))
        return old[0]

    def register_parts(self, reg):
        if not x86.RegisterExt.is_gpr(reg):
            raise MachineStop('unsupported', 'non-general-purpose register')
        number = x86.RegisterExt.number(x86.RegisterExt.full_register(reg))
        width = x86.RegisterExt.size(reg) * 8
        offset = 8 if reg in (x86.Register.AH, x86.Register.BH, x86.Register.CH, x86.Register.DH) else 0
        return number, offset, width

    def register(self, reg, value=None):
        number, offset, width = self.register_parts(reg)
        valid = self.register_known[number]
        if value is None and not all(valid[offset:offset + width]):
            raise MachineStop('undefined_input', 'read of an undefined register')
        low = self.ram_access(self.registers, 2 * number)
        high = self.ram_access(self.registers, 2 * number + 1)
        full = torch.cat((low, high))
        if value is None:
            return full[offset:offset + width]
        full[offset:offset + width] = value
        valid[offset:offset + width] = [True] * width
        if width == 32:
            full[32:] = 0
            valid[32:] = [True] * 32
        self.ram_access(self.registers, 2 * number, full[:32])
        self.ram_access(self.registers, 2 * number + 1, full[32:])

    def region(self, address, size, access):
        if size < 1 or address < 0 or address + size > 1 << 64:
            raise MachineStop('memory_fault', 'invalid memory range')
        for region in self.regions:
            if region.start <= address and address + size <= region.end:
                if not region.flags & access:
                    raise MachineStop('memory_fault', 'memory permission violation')
                return region, address - region.start
        raise MachineStop('memory_fault', 'unmapped memory range')

    def memory(self, address, size, value=None):
        region, offset = self.region(address, size, 4 if value is None else 2)
        if value is None:
            if not all(region.known[offset:offset + size]):
                raise MachineStop('undefined_input', 'read of undefined stack memory')
            return torch.cat([self.ram_access(region, offset + i) for i in range(size)])
        if len(value) != 8 * size:
            raise ValueError('memory value width mismatch')
        for i in range(size):
            self.ram_access(region, offset + i, value[8 * i:8 * (i + 1)])
            region.known[offset + i] = True

    def addition(self, a, b, carry=0):
        width = len(a)
        cin = torch.tensor([[carry]], dtype=self.dtype)
        if width == 64:
            low, cin = self.adders[32](a[None, :32], b[None, :32], cin)
            high, cout = self.adders[32](a[None, 32:], b[None, 32:], cin)
            return torch.cat((low[0], high[0])), bool(cout[0, 0])
        out, cout = self.adders[width](a[None], b[None], cin)
        return out[0], bool(cout[0, 0])

    def alu(self, name, a, b):
        if name in ('add', 'inc', 'sub', 'cmp', 'dec'):
            subtract = name in ('sub', 'cmp', 'dec')
            out, carry = self.addition(a, 1 - b if subtract else b, int(subtract))
            cf = not carry if subtract else carry
            of = bool((a[-1] != b[-1] if subtract else a[-1] == b[-1]) & (out[-1] != a[-1]))
        else:
            product = a * b
            out = {'and': product, 'test': product, 'or': a + b - product,
                   'xor': a + b - 2 * product}[name]
            cf = of = False
        old_cf = self.flags['b']
        self.flags = {'b': cf, 'o': of, 's': bool(out[-1]), 'e': not bool(out.any()),
                      'p': bool(out[:8].sum().remainder(2) == 0)}
        if name in ('inc', 'dec'):
            self.flags['b'] = old_cf
        return out

    def condition(self, suffix):
        aliases = {'z': 'e', 'nz': 'ne', 'c': 'b', 'nc': 'ae', 'na': 'be', 'nbe': 'a',
                   'nge': 'l', 'nl': 'ge', 'ng': 'le', 'nle': 'g', 'pe': 'p', 'po': 'np'}
        suffix = aliases.get(suffix, suffix)
        needed = {'a': ('b', 'e'), 'be': ('b', 'e'), 'l': ('s', 'o'), 'ge': ('s', 'o'),
                  'le': ('e', 's', 'o'), 'g': ('e', 's', 'o')}.get(suffix)
        if needed is None:
            key = suffix[1:] if suffix.startswith('n') else {'ae': 'b'}.get(suffix, suffix)
            if key not in self.flags:
                raise MachineStop('unsupported', 'unknown condition')
            needed = (key,)
        if any(self.flags[key] is None for key in needed):
            raise MachineStop('undefined_input', 'read of an undefined condition flag')
        f = self.flags
        if suffix in f:
            return f[suffix]
        if suffix in ('ne', 'no', 'ns', 'np', 'ae'):
            return not f[{'ae': 'b'}.get(suffix, suffix[1:])]
        return {'a': lambda: not f['b'] and not f['e'], 'be': lambda: f['b'] or f['e'],
                'l': lambda: f['s'] != f['o'], 'ge': lambda: f['s'] == f['o'],
                'le': lambda: f['e'] or f['s'] != f['o'],
                'g': lambda: not f['e'] and f['s'] == f['o']}[suffix]()

    def address(self, ins):
        if ins.segment_prefix in (x86.Register.FS, x86.Register.GS):
            raise MachineStop('unsupported', 'FS/GS addressing')
        if ins.is_ip_rel_memory_operand:
            return ins.ip_rel_memory_address
        out = self.bits(ins.memory_displacement, 64)
        for reg, scale in ((ins.memory_base, 1), (ins.memory_index, ins.memory_index_scale)):
            if reg == x86.Register.NONE:
                continue
            if not x86.RegisterExt.is_gpr64(reg):
                raise MachineStop('unsupported', 'non-64-bit address calculation')
            value = self.register(reg)
            shift = {1: 0, 2: 1, 4: 2, 8: 3}[scale]
            if shift:
                value = torch.cat((value.new_zeros(shift), value[:-shift]))
            out, _ = self.addition(out, value)
        return self.integer(out)

    def operand(self, ins, index, width, value=None):
        kind = ins.op_kind(index)
        if kind == x86.OpKind.REGISTER:
            return self.register(ins.op_register(index), value)
        if kind == x86.OpKind.MEMORY:
            return self.memory(self.address(ins), x86.MemorySizeExt.size(ins.memory_size), value)
        if kind in IMMEDIATES and value is None:
            return self.bits(ins.immediate(index), width)
        raise MachineStop('unsupported', 'operand encoding')

    def width(self, ins):
        if ins.op0_kind == x86.OpKind.REGISTER:
            return x86.RegisterExt.size(ins.op0_register) * 8
        if ins.op0_kind == x86.OpKind.MEMORY:
            return x86.MemorySizeExt.size(ins.memory_size) * 8
        raise MachineStop('unsupported', 'destination width')

    def push(self, value):
        sp, _ = self.addition(self.register(x86.Register.RSP), self.bits(-8, 64))
        self.register(x86.Register.RSP, sp)
        self.memory(self.integer(sp), 8, value)

    def pop(self):
        sp = self.register(x86.Register.RSP)
        value = self.memory(self.integer(sp), 8)
        sp, _ = self.addition(sp, self.bits(8, 64))
        self.register(x86.Register.RSP, sp)
        return value

    def syscall(self, next_ip):
        number = self.integer(self.register(x86.Register.RAX))
        self.register(x86.Register.RCX, self.bits(next_ip, 64))
        self.register_known[x86.RegisterExt.number(x86.Register.R11)] = [False] * 64
        if number in (60, 231):
            self.returncode = self.integer(self.register(x86.Register.RDI)) & 255
            raise MachineStop('ok', 'exit')
        if number not in (0, 1):
            raise MachineStop('unsupported', f'syscall {number}')
        fd, address, size = (self.integer(self.register(reg)) for reg in
                             (x86.Register.RDI, x86.Register.RSI, x86.Register.RDX))
        if size > 1 << 20:
            raise MachineStop('budget', 'I/O byte budget')
        if number == 0:
            if fd != 0:
                raise MachineStop('unsupported', 'read descriptor')
            data = self.stdin[self.stdin_offset:self.stdin_offset + size]
            if data:
                bits = encode_words(torch.tensor(list(data)), 8, dtype=self.dtype).reshape(-1)
                self.memory(address, len(data), bits)
            self.stdin_offset += len(data)
            count = len(data)
        else:
            if fd not in (1, 2):
                raise MachineStop('unsupported', 'write descriptor')
            bits = self.memory(address, size) if size else torch.empty(0, dtype=self.dtype)
            data = bytes(decode_words(bits.reshape(-1, 8)).tolist()) if size else b''
            output = self.stdout if fd == 1 else self.stderr
            if len(output) + len(data) > 1 << 20:
                raise MachineStop('budget', 'I/O byte budget')
            output.extend(data); count = size
        self.register(x86.Register.RAX, self.bits(count, 64))

    def shift(self, name, value, count):
        width = len(value); count &= 63 if width == 64 else 31
        if count == 0:
            return value
        fill = value[-1] if name == 'sar' else value.new_zeros(())
        index = torch.arange(width) - count if name in ('shl', 'sal') else torch.arange(width) + count
        valid = (index >= 0) & (index < width)
        out = torch.where(valid, value[index.clamp(0, width - 1)], fill)
        carry = (bool(value[width - count] if name in ('shl', 'sal') else value[count - 1])
                 if count < width else None)
        overflow = (bool(out[-1]) != carry if name in ('shl', 'sal') else
                    bool(value[-1]) if name == 'shr' else False) if count == 1 else None
        self.flags = {'b': carry, 'o': overflow, 's': bool(out[-1]), 'e': not bool(out.any()),
                      'p': bool(out[:8].sum().remainder(2) == 0)}
        return out

    def step(self, ins):
        name = MNEMONICS.get(ins.mnemonic, '?'); next_ip = ins.next_ip
        if name != 'endbr64' and (ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix):
            raise MachineStop('unsupported', 'instruction prefix')
        if name in ('nop', 'endbr64'):
            pass
        elif name == 'syscall':
            self.syscall(next_ip)
        elif name in ('jmp', 'call'):
            if ins.op0_kind != x86.OpKind.NEAR_BRANCH64:
                raise MachineStop('unsupported', 'indirect branch')
            if name == 'call':
                self.push(self.bits(next_ip, 64))
            next_ip = ins.near_branch_target
        elif name.startswith('j'):
            if ins.op0_kind != x86.OpKind.NEAR_BRANCH64:
                raise MachineStop('unsupported', 'branch encoding')
            if self.condition(name[1:]): next_ip = ins.near_branch_target
        elif name == 'ret' and ins.op_count == 0:
            next_ip = self.integer(self.pop())
        elif name == 'leave':
            self.register(x86.Register.RSP, self.register(x86.Register.RBP))
            self.register(x86.Register.RBP, self.pop())
        elif name in ('push', 'pop'):
            if name == 'push':
                value = self.operand(ins, 0, 64)
                if len(value) != 64: raise MachineStop('unsupported', 'non-64-bit push')
                self.push(value)
            else:
                if self.width(ins) != 64: raise MachineStop('unsupported', 'non-64-bit pop')
                self.operand(ins, 0, 64, self.pop())
        else:
            width = self.width(ins)
            if width not in (8, 16, 32, 64):
                raise MachineStop('unsupported', 'operand width')
            if name == 'lea':
                if ins.op1_kind != x86.OpKind.MEMORY:
                    raise MachineStop('unsupported', 'LEA operand')
                self.operand(ins, 0, width, self.bits(self.address(ins), width))
            elif name in ('mov', 'movzx', 'movsx', 'movsxd'):
                value = self.operand(ins, 1, width)
                if name == 'movzx':
                    value = torch.cat((value, value.new_zeros(width - len(value))))
                elif name in ('movsx', 'movsxd'):
                    value = torch.cat((value, value[-1].expand(width - len(value))))
                self.operand(ins, 0, width, value)
            elif name.startswith('set'):
                self.operand(ins, 0, width, self.bits(int(self.condition(name[3:])), width))
            elif name.startswith('cmov'):
                value = self.operand(ins, 1, width)
                if self.condition(name[4:]):
                    self.operand(ins, 0, width, value)
                elif width == 32:
                    number, _, _ = self.register_parts(ins.op0_register)
                    known = self.register_known[number][:32].copy()
                    low = self.ram_access(self.registers, 2 * number)
                    self.register(ins.op0_register, low)
                    self.register_known[number][:32] = known
            elif name in ('add', 'sub', 'and', 'or', 'xor', 'cmp', 'test', 'inc', 'dec', 'not', 'neg'):
                same = (ins.op_count == 2 and ins.op0_kind == ins.op1_kind == x86.OpKind.REGISTER
                        and ins.op0_register == ins.op1_register and name in ('xor', 'sub', 'cmp'))
                a = self.bits(0, width) if same else self.operand(ins, 0, width)
                if name == 'not': value = 1 - a
                elif name == 'neg': value = self.alu('sub', self.bits(0, width), a)
                else:
                    b = (self.bits(1, width) if name in ('inc', 'dec') else self.bits(0, width)
                         if same else self.operand(ins, 1, width))
                    value = self.alu(name, a, b)
                if name not in ('cmp', 'test'): self.operand(ins, 0, width, value)
            elif name in ('shl', 'sal', 'shr', 'sar'):
                value = self.operand(ins, 0, width)
                count = self.integer(self.operand(ins, 1, 8))
                self.operand(ins, 0, width, self.shift(name, value, count))
            else:
                raise MachineStop('unsupported', f'instruction {name}')
        self.ip = next_ip

    def load(self, raw):
        if not 64 <= len(raw) <= 1 << 20:
            raise MachineStop('unsupported', 'image capacity')
        elf = ELFFile(io.BytesIO(raw))
        if elf.elfclass != 64 or not elf.little_endian or elf.header.e_machine != 'EM_X86_64' or elf.header.e_type != 'ET_EXEC':
            raise MachineStop('unsupported', 'requires little-endian static x86-64 ET_EXEC')
        if not 1 <= elf.num_segments() <= 128:
            raise MachineStop('unsupported', 'ELF segment table capacity')
        self.regions = []
        for segment in elf.iter_segments():
            h = segment.header
            if h.p_type in ('PT_INTERP', 'PT_DYNAMIC'):
                raise MachineStop('unsupported', 'dynamic ELF')
            if h.p_type != 'PT_LOAD' or h.p_memsz == 0: continue
            if not 0 <= h.p_filesz <= h.p_memsz <= 4096 or h.p_offset + h.p_filesz > len(raw):
                raise MachineStop('unsupported', 'segment capacity or file span')
            if h.p_vaddr < 0 or h.p_vaddr + h.p_memsz > 1 << 64 or h.p_flags & 3 == 3:
                raise MachineStop('unsupported', 'segment address or writable executable memory')
            data = segment.data() + bytes(h.p_memsz - h.p_filesz)
            self.add_region(h.p_vaddr, h.p_flags, data, True)
        self.add_region(STACK, 6, bytes(4096), False)
        if sum(r.end - r.start for r in self.regions) > 16384:
            raise MachineStop('unsupported', 'total memory capacity')
        self.ip = int(elf.header.e_entry)
        self.region(self.ip, 1, 1)
        self.registers = Region(0, 6, self.register_ram, self.register_ram.empty(dtype=self.dtype), [True] * 32)
        self.register_known = [[False] * 64 for _ in range(16)]
        self.register(x86.Register.RSP, self.bits(STACK + 4096 - 16, 64))
        self.flags = dict.fromkeys(('b', 'o', 's', 'e', 'p'))

    def add_region(self, start, flags, data, known):
        if sum(r.end - r.start for r in self.regions) + len(data) > 16384:
            raise MachineStop('unsupported', 'total memory capacity')
        if any(start < r.end and r.start < start + len(data) for r in self.regions):
            raise MachineStop('unsupported', 'overlapping memory mappings')
        ram = ExactBitRAM(len(data), 8)
        state = encode_words(torch.tensor(list(data)), 8, dtype=self.dtype)[None]
        self.regions.append(Region(start, flags, ram, state, [known] * len(data), data if flags & 1 else None))

    @torch.inference_mode()
    def run(self, raw: bytes, case: Case = Case()):
        if not isinstance(raw, bytes): raise ValueError('raw ELF bytes required')
        binding = digest(json.dumps({'image': digest(raw), 'case': case.json(), 'spec': SPEC,
                                     'dtype': str(self.dtype), 'max_steps': self.max_steps}, sort_keys=True).encode())
        self.stdin, self.stdin_offset = case.stdin, 0
        self.stdout, self.stderr, self.returncode = bytearray(), bytearray(), None
        steps = 0; trace = []; histogram = {}; self.regions = []
        try:
            if case.args: raise MachineStop('unsupported', 'argv is not implemented')
            self.load(raw)
            while steps < self.max_steps:
                region, offset = self.region(self.ip, 1, 1)
                ins = x86.Decoder(64, region.code[offset:offset + 15], ip=self.ip).decode()
                if ins.is_invalid: raise MachineStop('unsupported', 'invalid instruction encoding')
                name = MNEMONICS.get(ins.mnemonic, '?')
                histogram[name] = histogram.get(name, 0) + 1
                steps += 1
                if len(trace) < self.trace_limit:
                    trace.append({'ip': self.ip, 'bytes': region.code[offset:offset + ins.len].hex(), 'text': str(ins)})
                self.step(ins)
            raise MachineStop('budget', 'instruction budget')
        except MachineStop as stop:
            status, reason = stop.status, stop.reason
        except (ValueError, EOFError, OSError, ELFError) as error:
            status, reason = 'invalid_artifact', str(error)
        state = hashlib.sha256()
        for region in self.regions:
            state.update(region.state.to(torch.uint8).numpy().tobytes())
        return {'status': status, 'reason': reason, 'returncode': self.returncode,
                'stdout': bytes(self.stdout).hex(), 'stderr': bytes(self.stderr).hex(),
                'image_sha256': digest(raw), 'state_binding_sha256': binding,
                'memory_sha256': state.hexdigest(), 'instructions': steps,
                'histogram': histogram, 'trace': trace, 'trace_truncated': steps > len(trace),
                'dtype': str(self.dtype), 'evidence': 'compiled_core', 'native_calls': 0,
                'reference_corrections': 0, 'spec': SPEC}
