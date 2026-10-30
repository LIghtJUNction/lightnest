"""Emit a reproducible JSON primitive probe; no learning or ELF claims.

From the repository root:
    PYTHONPATH=. uv run --locked python tools/probe_exact_core.py --steps 10000
The caller may redirect stdout into an artifact directory outside the checkout.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time

import torch

from archive.research.exact_core import ExactBitRAM, FixedWordAdder, decode_words, encode_words


def probe(steps: int, seed: int, dtype: torch.dtype) -> dict:
    rng = random.Random(seed)
    rows = 64
    ram = ExactBitRAM(rows, 32)
    state = ram.empty(dtype=dtype)
    reference = [0] * rows
    addresses = torch.eye(rows, dtype=dtype)
    read_errors = 0
    start = time.perf_counter()
    for _ in range(steps):
        address = rng.randrange(rows)
        value = rng.randrange(2**32)
        write = int(rng.random() < .75)
        state, old = ram(state, addresses[address:address + 1],
                         encode_words(torch.tensor([value]), dtype=dtype),
                         torch.tensor([[write]], dtype=dtype))
        read_errors += int(int(decode_words(old)[0]) != reference[address])
        if write:
            reference[address] = value
        # The reference is only compared. It never repairs or replaces state.
    final = decode_words(state[0]).tolist()
    final_errors = sum(a != b for a, b in zip(final, reference, strict=True))
    ram_seconds = time.perf_counter() - start
    pairs = [(rng.randrange(2**32), rng.randrange(2**32), rng.randrange(2)) for _ in range(4096)]
    a = torch.tensor([x[0] for x in pairs])
    b = torch.tensor([x[1] for x in pairs])
    c = torch.tensor([[x[2]] for x in pairs], dtype=dtype)
    adder = FixedWordAdder(32).to(dtype=dtype)
    result, carry = adder(encode_words(a, dtype=dtype), encode_words(b, dtype=dtype), c)
    actual = decode_words(result).tolist()
    carries = carry[:, 0].to(torch.int64).tolist()
    add_errors = sum(s != (x + y + cin) % 2**32 or cout != (x + y + cin) // 2**32
                     for (x, y, cin), s, cout in zip(pairs, actual, carries, strict=True))
    return {"dtype": str(dtype), "ram_rows": rows, "word_bits": 32,
            "ram_steps_without_reference_correction": steps,
            "ram_read_mismatches": read_errors, "ram_final_word_mismatches": final_errors,
            "ram_probe_seconds_including_validation_and_python_reference": ram_seconds,
            "physical_state_bytes": state.numel() * state.element_size(),
            "logical_state_bytes": rows * 4,
            "addition_cases": len(pairs), "addition_mismatches": add_errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000000:
        parser.error('--steps must be in 1..1000000')
    torch.set_num_threads(1)
    result = {"scope": "exact_tensor_primitives_only_not_cpu_or_model_evaluation",
              "python": platform.python_version(), "torch": torch.__version__,
              "device": "cpu", "seed": args.seed,
              "results": [probe(args.steps, args.seed, dtype) for dtype in (torch.float32, torch.bfloat16)]}
    print(json.dumps(result, indent=2))
    if any(r["ram_read_mismatches"] or r["ram_final_word_mismatches"] or r["addition_mismatches"]
           for r in result["results"]):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
