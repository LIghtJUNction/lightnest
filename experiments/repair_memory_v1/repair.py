"""Keep a small number of incomplete candidates backed by visible progress.

First finish the unchanged one-edit search. Only if it abstains, retain up to
8 candidates that passed a strictly longer prefix of the original test suite.
Each is paired with its first remaining failure. A second edit must touch code
read on that failing run. Search these branches round-robin, then use remaining budget for the same
random-pair stream as the control; commit only after
ALL original visible tests pass. No patch is applied to a user file.

This is a bounded counterexample-guided search adaptation, not a new neural
architecture, universal correctness proof, or claim of global method novelty.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import random
import struct
from experiments.compiled_v3.search import Request, edits, patched
from experiments.compiled_v3.vm import NativeVM, pack


@dataclass(frozen=True)
class Result:
    code: bytes
    status: str
    patches: tuple[tuple[int, int], ...]
    tried: int
    vm_calls: int
    vm_steps: int
    kept: int
    assurance: str = 'sample_checked_not_proven'


def candidates(code, coverage=None):
    options = list(edits(code))
    options.sort(key=lambda p: hashlib.sha256(code + struct.pack('<II', *p)).digest())
    return [p for p in options if coverage is None or p[0] // 4 in coverage]


def repair_bridge(request: Request, vm: NativeVM, *, mode='bridge', budget=4096,
                  width=8) -> Result:
    if mode not in ('single', 'random_pair', 'random_memory', 'bridge_only', 'bridge'):
        raise ValueError('Unknown search mode')
    if type(budget) is not int or type(width) is not int or not 1 <= budget <= 100000 or not 1 <= width <= 32:
        raise ValueError('Invalid bounded search budget')
    before = tuple(vm.work)
    base, entry = request.code, request.entry
    packed = pack(request.cases, request.expected)
    tried, kept = 0, []
    def result(code, status, patches=()):
        return Result(code, status, patches, tried,
            int(vm.work[0]-before[0]), int(vm.work[1]-before[1]), len(kept))
    failure = vm.check(base, entry, packed)
    if failure < 0:
        return result(base, 'unchanged')
    coverage, failures = None, 0
    for inp, expected in zip(request.cases[failure:], request.expected[failure:]):
        out, seen = vm.run(base, entry, inp, trace=True)
        if out.observable() != expected.observable():
            coverage = seen if coverage is None else coverage & seen
            failures += 1
            if failures == 4:
                break
    single = candidates(base, coverage)
    for index, change in enumerate(single[:budget]):
        code = patched(base, *change)
        tried += 1
        remaining = vm.check(code, entry, packed)
        if remaining < 0:
            return result(code, 'candidate', (change,))
        progress = remaining > failure
        if mode == 'random_memory' or (mode in ('bridge', 'bridge_only') and progress):
            kept.append((remaining, index, change, code))
            if mode == 'random_memory':
                # Same memory size and second-stage path filter; retain failures
                # without selecting for demonstrated progress.
                kept.sort(key=lambda x: hashlib.sha256(
                    base + struct.pack('<II', *x[2]) + b'memory-control').digest())
            else:
                kept.sort(key=lambda x: (-x[0], x[1]))
            del kept[width:]
    if tried >= budget or mode == 'single':
        return result(base, 'budget' if tried >= budget else 'abstain')
    def random_fallback(exclude):
        # Equal two-edit capability and candidate budget, no progress retention.
        options = candidates(base)
        rng = random.Random(int.from_bytes(hashlib.sha256(base).digest()[:8], 'little'))
        nonlocal tried
        seen_pairs = set(exclude)
        for _ in range(20 * budget):
            if tried >= budget or len(options) < 2:
                break
            a, b = rng.sample(options, 2)
            if a[0] == b[0]:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            code = patched(patched(base, *a), *b)
            tried += 1
            if vm.check(code, entry, packed) < 0:
                return result(code, 'candidate', pair)
        return result(base, 'budget' if tried >= budget else 'abstain')
    if mode == 'random_pair':
        return random_fallback(())
    branches = []
    for remaining, index, first, code in kept:
        _, seen = vm.run(code, entry, request.cases[remaining], trace=True)
        # Do not overwrite the first edit; original bytes are the common base.
        second = [p for p in candidates(code, seen) if p[0] != first[0]]
        branches.append((first, code, iter(second)))
    tested = set()
    while branches and tried < budget:
        active = []
        for first, partial, stream in branches:
            if tried >= budget:
                break
            second = next(stream, None)
            if second is None:
                continue
            active.append((first, partial, stream))
            pair = tuple(sorted((first, second)))
            if pair in tested:
                continue
            tested.add(pair)
            code = patched(partial, *second)
            tried += 1
            if vm.check(code, entry, packed) < 0:
                return result(code, 'candidate', pair)
        branches = active
    if mode in ('bridge', 'random_memory') and tried < budget:
        return random_fallback(tested)
    return result(base, 'budget' if tried >= budget else 'abstain')
