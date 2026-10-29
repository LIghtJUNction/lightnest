"""Opt-in exact tensor primitives; NOT an ELF executor or a trained capability.

The RAM specializes the delta rule to a one-hot key, alpha=1 and beta in {0, 1}.
The adder uses fixed linear/ReLU gates on individual bits, not floating-point
representations of whole integers. State is explicit, never global or implicit.
No existing RGB model, training path, checkpoint, or native runner is changed.

Concept reference: divagr18/qwenputer (see docs/QWENPUTER_ADAPTATION.md).
Implementation below is original and does not vendor upstream code.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_DTYPES = (torch.float32, torch.bfloat16)


def _positive_int(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _binary(value: Tensor, name: str) -> None:
    if value.dtype not in _DTYPES:
        raise ValueError(f"{name} must use float32 or bfloat16")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain exact binary values, not soft probabilities")


def encode_words(words: Tensor, width: int = 32, *, dtype=torch.float32) -> Tensor:
    """Boundary encoding only: uint values in an int64 tensor -> little-endian bits.

    Native integer operations here encode input; they do not perform the adder's
    computation. Width <= 32 leaves room for carries in the reference tests.
    """
    _positive_int(width, "width", 32)
    if dtype not in _DTYPES or words.dtype != torch.int64 or words.ndim != 1:
        raise ValueError("encoding needs an int64 vector and float32/bfloat16 output")
    if words.numel() == 0 or bool(((words < 0) | (words >= (1 << width))).any()):
        raise ValueError("word outside the unsigned input range")
    shifts = torch.arange(width, device=words.device, dtype=torch.int64)
    return ((words[:, None] >> shifts) & 1).to(dtype)


def decode_words(bits: Tensor) -> Tensor:
    """Boundary decoding only; not used to compute RAM updates or sums."""
    if bits.ndim != 2 or bits.shape[0] == 0 or not 1 <= bits.shape[1] <= 32:
        raise ValueError("decoding needs a nonempty [batch, 1..32] bit matrix")
    _binary(bits, "bits")
    shifts = torch.arange(bits.shape[1], device=bits.device, dtype=torch.int64)
    return (bits.to(torch.int64) << shifts).sum(-1)


class ExactBitRAM(nn.Module):
    """Validated, out-of-place GDN-style RAM transition with read-before-write.

    state: [B, R, W]; key: [B, R] exactly one-hot;
    value: [B, W] bits; write_enable: [B, 1] exactly 0 or 1.

    Returns (new_state, old_row). Neither input state nor any module buffer is
    mutated. The caller owns state lifetime and must invalidate/reboot it after
    changes to the artifact, input, machine specification, or execution policy.

    This deliberately dense prototype is a correctness baseline, not an
    efficient replacement for native RAM. Validation also scans state.
    """
    def __init__(self, rows: int = 64, word_bits: int = 32):
        super().__init__()
        _positive_int(rows, "rows", 4096)
        _positive_int(word_bits, "word_bits", 32)
        self.rows, self.word_bits = rows, word_bits

    def empty(self, batch: int = 1, *, dtype=torch.float32, device=None) -> Tensor:
        _positive_int(batch, "batch", 1024)
        if dtype not in _DTYPES:
            raise ValueError("unsupported state dtype")
        return torch.zeros(batch, self.rows, self.word_bits, dtype=dtype, device=device)

    def forward(self, state: Tensor, key: Tensor, value: Tensor,
                write_enable: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim != 3 or state.shape[0] == 0:
            raise ValueError("state must have shape [batch, rows, word_bits]")
        batch = state.shape[0]
        shapes = ((state, (batch, self.rows, self.word_bits), "state"),
                  (key, (batch, self.rows), "key"),
                  (value, (batch, self.word_bits), "value"),
                  (write_enable, (batch, 1), "write_enable"))
        for tensor, shape, name in shapes:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"wrong {name} shape")
            if tensor.dtype != state.dtype or tensor.device != state.device:
                raise ValueError("RAM inputs must share dtype and device")
            _binary(tensor, name)
        if not bool((key.to(torch.int64).sum(-1) == 1).all()):
            raise ValueError("key must select exactly one row")
        # alpha is exactly 1, not sigmoid(large_number). beta is write_enable.
        # With a one-hot key and binary state all intermediate values are small.
        old_row = torch.bmm(key[:, None, :], state).squeeze(1)
        delta = write_enable * (value - old_row)
        new_state = state + key[:, :, None] * delta[:, None, :]
        return new_state, old_row


class FixedWordAdder(nn.Module):
    """Unsigned ripple-carry adder compiled into fixed linear/ReLU gates.

    Input/output words are bit vectors. For s=a+b+c (s in {0,1,2,3}):
        carry = relu(s-1) - relu(s-2)
        sum_bit = s - 2*carry
    Each bit uses two fixed linear maps and one ReLU. The Python loop has a
    constant circuit depth; it does not branch on operand values. No integer
    arithmetic, emulator, lookup oracle, or subprocess computes the sum.

    The maps are buffers, not trainable parameters. This is one arithmetic
    primitive, not an ISA, CPU, Qwen merge, or learned execution predictor.
    """
    def __init__(self, word_bits: int = 32):
        super().__init__()
        _positive_int(word_bits, "word_bits", 32)
        self.word_bits = word_bits
        self.register_buffer("encode_weight", torch.ones(3, 3))
        self.register_buffer("encode_bias", torch.tensor([0., -1., -2.]))
        self.register_buffer("decode_weight", torch.tensor([[1., -2., 2.], [0., 1., -1.]]))

    def forward(self, a: Tensor, b: Tensor, carry_in: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != self.word_bits or b.shape != a.shape:
            raise ValueError("operands must have shape [batch, word_bits]")
        if carry_in is None:
            carry_in = a.new_zeros(a.shape[0], 1)
        if tuple(carry_in.shape) != (a.shape[0], 1):
            raise ValueError("carry_in must have shape [batch, 1]")
        for tensor, name in ((a, "a"), (b, "b"), (carry_in, "carry_in")):
            if tensor.dtype != self.encode_weight.dtype or tensor.device != self.encode_weight.device:
                raise ValueError("adder buffers and inputs must share dtype and device")
            _binary(tensor, name)
        carry = carry_in
        output = []
        for bit in range(self.word_bits):
            inputs = torch.cat((a[:, bit:bit + 1], b[:, bit:bit + 1], carry), -1)
            hidden = F.relu(F.linear(inputs, self.encode_weight, self.encode_bias))
            gates = F.linear(hidden, self.decode_weight)
            output.append(gates[:, :1])
            carry = gates[:, 1:2]
        return torch.cat(output, -1), carry
