"""Bounded real x86-64 byte execution as a tensor-resident module.

Implements an intentionally small instruction/syscall subset. No host-side
instruction decoder or external emulator is called during the fixed rollout.
"""
from __future__ import annotations

import torch
from torch import nn

RUNNING, EXITED, UNSUPPORTED, FAULT = range(4)


class X86Internal(nn.Module):
    """Run real x86-64 machine bytes from one fixed, contiguous virtual mapping.

    Supported: B8..BF mov r32,imm32; 89 /r mov r32,r32; 31 /r xor r32,r32; 48 8d 35 disp32
    lea rsi,[rip+disp32]; 83 c0 imm8 add eax,imm8; EB rel8 jmp; 74 rel8
    jz; 90 nop; 0f05 syscall with Linux x86-64 write(1/2) and exit(60).
    An optional learned module approximates openat(257) only; it is not a filesystem.
    Only ZF is modeled. Memory is read-only to the program; no stack or argv.
    Execution steps are fixed, batched tensor operations on the module device.
    """

    def __init__(self, memory_size: int = 4096, output_size: int = 64, base: int = 0x400000, environment: nn.Module | None = None):
        super().__init__()
        if not 8 <= memory_size <= 65536 or not 1 <= output_size <= 1024 or base < 0:
            raise ValueError('invalid machine dimensions')
        self.memory_size, self.output_size, self.base = memory_size, output_size, base
        self.environment = environment
        self.register_buffer('fetch_offsets', torch.arange(7, dtype=torch.long), persistent=False)
        self.register_buffer('output_offsets', torch.arange(output_size, dtype=torch.long), persistent=False)
        self.register_buffer('open_returns', torch.tensor([3, -2, -13], dtype=torch.long), persistent=False)
        if environment is not None:
            self.register_buffer('path_offsets', torch.arange(environment.path_size, dtype=torch.long), persistent=False)

    def forward(self, memory: torch.Tensor, readable: torch.Tensor, executable: torch.Tensor,
                entry: torch.Tensor, *, steps: int):
        if memory.ndim != 2 or memory.shape[1] != self.memory_size:
            raise ValueError('memory must have shape [batch, memory_size]')
        batch = memory.shape[0]
        if readable.shape != memory.shape or executable.shape != memory.shape or entry.shape != (batch,):
            raise ValueError('mapping or entry shape mismatch')
        if memory.dtype != torch.long or entry.dtype != torch.long or readable.dtype != torch.bool or executable.dtype != torch.bool:
            raise ValueError('memory and entry must be int64; maps must be boolean')
        if any(x.device != self.fetch_offsets.device for x in (memory, readable, executable, entry)):
            raise ValueError('model and inputs must be on the same device')
        if not 1 <= steps <= 1024:
            raise ValueError('steps must be between 1 and 1024')
        regs = torch.zeros((batch, 16), dtype=torch.long, device=memory.device)
        reg_valid = torch.zeros((batch, 16), dtype=torch.bool, device=memory.device)
        rip = entry.clone()
        invalid_bytes = ((memory < 0) | (memory > 255)).any(dim=1)
        status = torch.where(invalid_bytes, FAULT, RUNNING).long()
        zf = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        zf_valid = torch.zeros_like(zf)
        output = torch.zeros((batch, 2, self.output_size), dtype=torch.long, device=memory.device)
        lengths = torch.zeros((batch, 2), dtype=torch.long, device=memory.device)
        exit_code = torch.zeros(batch, dtype=torch.long, device=memory.device)
        open_logits, open_masks = [], []
        for _ in range(steps):
            active = status.eq(RUNNING)
            positions = rip[:, None] + self.fetch_offsets
            valid_positions = (positions >= 0) & (positions < self.memory_size)
            safe_positions = positions.clamp(0, self.memory_size - 1)
            code = memory.gather(1, safe_positions)
            code_exec = executable.gather(1, safe_positions) & valid_positions
            op, b1, b2, b3, b4, b5, b6 = code.unbind(-1)
            mov = (op >= 0xb8) & (op <= 0xbf)
            xor = op.eq(0x31) & (b1 >= 0xc0)
            mov_reg = op.eq(0x89) & (b1 >= 0xc0)
            lea = op.eq(0x48) & b1.eq(0x8d) & b2.eq(0x35)
            add = op.eq(0x83) & b1.eq(0xc0)
            jmp = op.eq(0xeb)
            jz = op.eq(0x74)
            nop = op.eq(0x90)
            syscall = op.eq(0x0f) & b1.eq(0x05)
            known = mov | mov_reg | xor | lea | add | jmp | jz | nop | syscall
            size = torch.where(mov, 5, torch.where(lea, 7, torch.where(add, 3,
                    torch.where(xor | mov_reg | jmp | jz | syscall, 2, 1))))
            fetch_ok = (code_exec | (self.fetch_offsets[None] >= size[:, None])).all(dim=1)
            arith = b1 + (b2 << 8) + (b3 << 16) + (b4 << 24)
            displacement = b3 + (b4 << 8) + (b5 << 16) + (b6 << 24)
            displacement = torch.where(displacement >= (1 << 31), displacement - (1 << 32), displacement)
            source = ((b1 >> 3) & 7).clamp(0, 7)
            destination = (b1 & 7).clamp(0, 7)
            src_value = regs.gather(1, source[:, None]).squeeze(1) & 0xffffffff
            dst_value = regs.gather(1, destination[:, None]).squeeze(1) & 0xffffffff
            xor_value = src_value ^ dst_value
            src_defined = reg_valid.gather(1, source[:, None]).squeeze(1)
            dst_defined = reg_valid.gather(1, destination[:, None]).squeeze(1)
            xor_defined = (source == destination) | (src_defined & dst_defined)
            sign8 = torch.where(b2 >= 128, b2 - 256, b2)
            add_value = (regs[:, 0] + sign8) & 0xffffffff
            sys_write = syscall & regs[:, 0].eq(1)
            sys_exit = syscall & regs[:, 0].eq(60)
            sys_open = syscall & regs[:, 0].eq(257)
            if self.environment is not None:
                path_addr = regs[:, 6, None] - self.base + self.path_offsets
                path_in_range = (path_addr >= 0) & (path_addr < self.memory_size)
                path_safe = path_addr.clamp(0, self.memory_size - 1)
                path_bytes = memory.gather(1, path_safe)
                path_readable = readable.gather(1, path_safe) & path_in_range
                before_zero = torch.cat((torch.ones_like(path_bytes[:, :1]), path_bytes.ne(0).long().cumprod(dim=1)[:, :-1]), dim=1).bool()
                path_ok = ((~before_zero) | path_readable).all(dim=1) & (before_zero & path_bytes.eq(0) & path_readable).any(dim=1)
                logits = self.environment(path_bytes, regs[:, 2])
                open_value = self.open_returns[logits.argmax(dim=1)]
                open_logits.append(logits)
                open_masks.append(active & sys_open)
            else:
                path_ok = torch.zeros_like(active)
                open_value = torch.zeros_like(rip)
            fd = regs[:, 7]
            fd_ok = fd.eq(1) | fd.eq(2)
            fd_index = (fd - 1).clamp(0, 1)
            previous_length = lengths.gather(1, fd_index[:, None]).squeeze(1)
            count = regs[:, 2]
            start = regs[:, 6] - self.base
            address = start[:, None] + self.output_offsets
            read_pos_ok = (address >= 0) & (address < self.memory_size)
            safe_read = address.clamp(0, self.memory_size - 1)
            can_read = readable.gather(1, safe_read) & read_pos_ok
            access_ok = ((self.output_offsets[None] >= count[:, None]) | can_read).all(dim=1)
            write_ok = fd_ok & (count >= 0) & (count <= self.output_size - previous_length) & access_ok
            invalid_branch = jz & ~zf_valid
            invalid_register = ((xor & ~xor_defined) | (mov_reg & ~src_defined) | (add & ~reg_valid[:, 0])
                                | (syscall & ~reg_valid[:, 0])
                                | (sys_write & ~(reg_valid[:, 7] & reg_valid[:, 6] & reg_valid[:, 2]))
                                | (sys_exit & ~reg_valid[:, 7])
                                | (sys_open & ~(reg_valid[:, 7] & reg_valid[:, 6] & reg_valid[:, 2])))
            invalid_memory = ~fetch_ok | (sys_write & ~write_ok) | invalid_branch | invalid_register
            if self.environment is not None:
                invalid_memory = invalid_memory | (sys_open & ~path_ok)
            unsupported = ~known | (syscall & ~(sys_write | sys_exit | (sys_open & (self.environment is not None))))
            execute = active & ~invalid_memory & ~unsupported
            status = torch.where(active & invalid_memory, FAULT,
                     torch.where(active & ~invalid_memory & unsupported, UNSUPPORTED, status))
            register_index = torch.where(mov, op - 0xb8, torch.where(xor | mov_reg, destination,
                             torch.where(lea, 6, 0))).clamp(0, 15)
            value = torch.where(mov, arith, torch.where(xor, xor_value, torch.where(mov_reg, src_value,
                    torch.where(lea, self.base + rip + 7 + displacement,
                    torch.where(add, add_value, torch.where(sys_open, open_value, count))))))
            write_register = execute & (mov | mov_reg | xor | lea | add | sys_write | (sys_open & (self.environment is not None)))
            changed_regs = regs.scatter(1, register_index[:, None], value[:, None])
            regs = torch.where(write_register[:, None], changed_regs, regs)
            changed_valid = reg_valid.scatter(1, register_index[:, None], torch.ones((batch, 1), dtype=torch.bool, device=memory.device))
            reg_valid = torch.where(write_register[:, None], changed_valid, reg_valid)
            # SYSCALL clobbers RCX/R11; other flags saved to R11 are not modeled.
            syscall_done = execute & syscall
            rcx_values = (self.base + rip + 2)[:, None]
            with_rcx = regs.scatter(1, torch.ones((batch, 1), dtype=torch.long, device=memory.device), rcx_values)
            regs = torch.where(syscall_done[:, None], with_rcx, regs)
            rcx_valid = reg_valid.scatter(1, torch.ones((batch, 1), dtype=torch.long, device=memory.device), torch.ones((batch, 1), dtype=torch.bool, device=memory.device))
            reg_valid = torch.where(syscall_done[:, None], rcx_valid, reg_valid)
            changed_zf = torch.where(xor, xor_value.eq(0), add_value.eq(0))
            sets_zf = execute & (xor | add)
            zf = torch.where(sets_zf, changed_zf, zf)
            zf_valid = zf_valid | sets_zf
            buffer = output.gather(1, fd_index[:, None, None].expand(-1, 1, self.output_size)).squeeze(1)
            out_pos = self.output_offsets[None] - previous_length[:, None]
            out_addr = (start[:, None] + out_pos).clamp(0, self.memory_size - 1)
            data = memory.gather(1, out_addr)
            append_mask = (out_pos >= 0) & (out_pos < count[:, None])
            changed_buffer = torch.where(append_mask, data, buffer)
            changed_output = output.scatter(1, fd_index[:, None, None].expand(-1, 1, self.output_size), changed_buffer[:, None])
            changed_lengths = lengths.scatter(1, fd_index[:, None], (previous_length + count)[:, None])
            writing = execute & sys_write
            output = torch.where(writing[:, None, None], changed_output, output)
            lengths = torch.where(writing[:, None], changed_lengths, lengths)
            exit_code = torch.where(execute & sys_exit, fd & 255, exit_code)
            status = torch.where(execute & sys_exit, EXITED, status)
            rel8 = torch.where(b1 >= 128, b1 - 256, b1)
            next_rip = torch.where(jmp | (jz & zf), rip + 2 + rel8, rip + size)
            rip = torch.where(execute & ~sys_exit, next_rip, rip)
        result = {'registers': regs, 'register_valid': reg_valid, 'rip': rip, 'status': status, 'zero_flag': zf,
                  'output': output, 'output_lengths': lengths, 'exit_code': exit_code}
        if self.environment is not None:
            result['open_logits'] = torch.stack(open_logits, dim=1)
            result['open_mask'] = torch.stack(open_masks, dim=1)
        return result
