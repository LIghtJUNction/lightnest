"""Seeded CPU stress and closed-loop probes, not a CPU or model benchmark.

Run with PYTHONPATH=. and redirect JSON outside the source checkout.
The Python reference observes results but never repairs the tensor state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from lightnest_rgb.exact_core import ExactBitRAM, FixedWordAdder, decode_words, encode_words
from tools.probe_exact_core import probe


@torch.inference_mode()
def addition_probe(cases: int, seed: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator().manual_seed(seed)
    adder = FixedWordAdder(32).to(dtype=dtype)
    digest = hashlib.sha256()
    errors = 0
    start = time.perf_counter()
    for offset in range(0, cases, 4096):
        size = min(4096, cases - offset)
        a = torch.randint(2**32, (size,), generator=generator)
        b = torch.randint(2**32, (size,), generator=generator)
        c = torch.randint(2, (size, 1), generator=generator)
        for value in (a, b, c):
            digest.update(value.numpy().astype('<i8', copy=False).tobytes())
        result, carry = adder(encode_words(a, dtype=dtype), encode_words(b, dtype=dtype), c.to(dtype))
        expected = a + b + c[:, 0]
        errors += int(((decode_words(result) != expected % 2**32)
                       | (carry[:, 0].long() != expected // 2**32)).sum())
    return {'cases': cases, 'mismatches': errors, 'input_sha256': digest.hexdigest(),
            'seconds_including_encoding_validation_reference': time.perf_counter() - start}


@torch.inference_mode()
def feedback_probe(steps: int, seed: int, dtype: torch.dtype, lanes: int = 8) -> dict:
    """Rotate over eight rows, feeding each new sum and carry into later steps.

    Each lane has independent RAM and carry. Addresses follow a frozen schedule;
    increments are external seeded inputs. This is not learned control or an ISA.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = 8
    ram = ExactBitRAM(rows, 32)
    adder = FixedWordAdder(32).to(dtype=dtype)
    state = ram.empty(lanes, dtype=dtype)
    carry = torch.zeros(lanes, 1, dtype=dtype)
    zero = torch.zeros(lanes, 32, dtype=dtype)
    read_gate = torch.zeros(lanes, 1, dtype=dtype)
    write_gate = torch.ones(lanes, 1, dtype=dtype)
    keys = torch.eye(rows, dtype=dtype)
    reference = torch.zeros(lanes, rows, dtype=torch.int64)
    reference_carry = torch.zeros(lanes, dtype=torch.int64)
    read_errors = result_errors = 0
    digest = hashlib.sha256()
    start = time.perf_counter()
    for step in range(steps):
        row = step % rows
        key = keys[row:row + 1].expand(lanes, -1)
        state, old = ram(state, key, zero, read_gate)
        increment = torch.randint(2**32, (lanes,), generator=generator)
        digest.update(increment.numpy().astype('<i8', copy=False).tobytes())
        result, carry = adder(old, encode_words(increment, dtype=dtype), carry)
        state, _ = ram(state, key, result, write_gate)
        # Integer calculations below are an independent observer only.
        read_errors += int((decode_words(old) != reference[:, row]).sum())
        expected = reference[:, row] + increment + reference_carry
        reference[:, row] = expected % 2**32
        reference_carry = expected // 2**32
        result_errors += int(((decode_words(result) != reference[:, row])
                              | (carry[:, 0].long() != reference_carry)).sum())
    final = decode_words(state.reshape(lanes * rows, 32)).reshape(lanes, rows)
    return {'steps_per_lane': steps, 'lanes': lanes, 'word_updates': steps * lanes,
            'read_mismatches': read_errors, 'result_or_carry_mismatches': result_errors,
            'final_word_mismatches': int((final != reference).sum()),
            'input_sha256': digest.hexdigest(),
            'final_words_sha256': hashlib.sha256(final.numpy().astype('<i8', copy=False).tobytes()).hexdigest(),
            'seconds_including_validation_reference': time.perf_counter() - start,
            'reference_corrections': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ram-steps', type=int, default=100000)
    parser.add_argument('--addition-cases', type=int, default=131072)
    parser.add_argument('--feedback-steps', type=int, default=2048)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261029, 20261030, 20261031])
    args = parser.parse_args()
    for name in ('ram_steps', 'addition_cases', 'feedback_steps'):
        if not 1 <= getattr(args, name) <= 1000000:
            parser.error(f'{name} must be in 1..1000000')
    if not 1 <= len(args.seeds) <= 16 or any(not 0 <= s < 2**63 for s in args.seeds):
        parser.error('provide 1..16 seeds in 0..2**63-1')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in
                     ('lightnest_rgb/exact_core.py', 'tools/probe_exact_core.py', 'tools/stress_exact_core.py')}
    results = []
    started = datetime.now(timezone.utc).isoformat()
    with torch.inference_mode():
        for seed in args.seeds:
            for dtype in (torch.float32, torch.bfloat16):
                results.append({'seed': seed, 'dtype': str(dtype),
                                'ram': probe(args.ram_steps, seed, dtype),
                                'addition': addition_probe(args.addition_cases, seed, dtype),
                                'feedback': feedback_probe(args.feedback_steps, seed, dtype)})
    failed = any(r['ram']['ram_read_mismatches'] or r['ram']['ram_final_word_mismatches']
                 or r['ram']['addition_mismatches'] or r['addition']['mismatches']
                 or r['feedback']['read_mismatches'] or r['feedback']['result_or_carry_mismatches']
                 or r['feedback']['final_word_mismatches'] for r in results)
    print(json.dumps({'scope': 'primitives_and_fixed_host_scheduled_recurrence_only',
                      'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                      'python': platform.python_version(), 'torch': torch.__version__,
                      'device': 'cpu', 'threads': torch.get_num_threads(),
                      'source_sha256': source_hashes, 'config': vars(args),
                      'passed': not failed, 'results': results}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
