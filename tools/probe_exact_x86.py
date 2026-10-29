"""Freeze and differentially evaluate a bounded tensor x86 ELF interpreter.

Only locally authored C/assembly programs are compiled. Native answers are
obtained independently in the existing Linux sandbox, never fed to the core.
All binaries and raw results must stay outside the source checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lightnest_rgb.artifact import digest
from lightnest_rgb.corpus import compile_generation_target, external
from lightnest_rgb.exact_x86 import ExactX86Runner, SPEC
from lightnest_rgb.exact_core import FixedWordAdder
from lightnest_rgb.execute import Case, ExecutableRunner, same_result

FAMILIES = {
    'arithmetic32': (32, 'return (a + b) ^ (a - b);'),
    'arithmetic64': (64, 'return (a + b) ^ (a - b);'),
    'logic32': (32, 'return ((a & b) | (a ^ b)) ^ (~a);'),
    'unsigned_select': (32, 'return a > b ? a - b : b - a;'),
    'signed_select': (32, 'return (int)a > (int)b ? a : b;'),
    'loop': (32, 'U s=a; for(U i=0; i<(b&7); ++i) s=(s<<1)^(s>>3)^i; return s;'),
    'memory': (32, 'volatile U s[4]; s[0]=a; s[1]=b; s[2]=a+b; s[3]=a^b; return s[(a>>29)&3]+s[(b>>29)&3];'),
    'call': (32, 'return inner(a,b)+inner(b,a);'),
}


def source_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in
            ('lightnest_rgb/exact_core.py', 'lightnest_rgb/exact_x86.py',
             'lightnest_rgb/execute.py', 'tools/probe_exact_x86.py', 'uv.lock')}


def wrapper(width):
    reg, size = ('eax', 4) if width == 32 else ('rax', 8)
    load = f'mov input(%rip),%edi\n mov input+{size}(%rip),%esi' if width == 32 else f'mov input(%rip),%rdi\n mov input+{size}(%rip),%rsi'
    return f'''.global _start
.text
_start:
 lea private_stack_top(%rip),%rsp
 xor %ebp,%ebp
 xor %ebx,%ebx
 xor %r12d,%r12d
 xor %r13d,%r13d
 xor %r14d,%r14d
 xor %r15d,%r15d
 xor %eax,%eax
 xor %edi,%edi
 lea input(%rip),%rsi
 mov ${2 * size},%edx
 syscall
 {load}
 call operation
 mov %{reg},output(%rip)
 mov $1,%eax
 mov $1,%edi
 lea output(%rip),%rsi
 mov ${size},%edx
 syscall
 xor %edi,%edi
 mov $60,%eax
 syscall
.bss
.balign 8
input: .skip {2 * size}
output: .skip {size}
.balign 16
private_stack: .skip 3968
private_stack_top:
.section .note.GNU-stack,"",@progbits
'''


def build_program(directory, family, compiler='gcc', optimization='O0'):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    width, body = FAMILIES[family]
    ctype = 'unsigned int' if width == 32 else 'unsigned long'
    source = (f'typedef {ctype} U;\n'
              '__attribute__((noinline)) U inner(U a,U b){ return (a+b)^(a-b); }\n'
              f'__attribute__((noinline)) U operation(U a,U b){{ {body} }}\n')
    (directory / 'operation.c').write_text(source)
    (directory / 'start.s').write_text(wrapper(width))
    output = directory / 'program.elf'
    command = [compiler, f'-{optimization}', '-ffreestanding', '-fno-pie', '-fno-stack-protector',
               '-fno-asynchronous-unwind-tables', '-fno-unwind-tables', '-fno-tree-vectorize',
               '-fno-tree-slp-vectorize', '-nostdlib', '-static', '-no-pie', '-Wl,--build-id=none',
               '-Wl,-e,_start', str(directory / 'start.s'), str(directory / 'operation.c'), '-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode: raise RuntimeError(result.stderr)
    return {'family': family, 'width': width, 'compiler': compiler, 'optimization': optimization,
            'path': str(output), 'sha256': digest(output.read_bytes()),
            'source_sha256': digest(source.encode()), 'compile_command': command,
            'compiler_stderr': result.stderr}


def inputs(width, seeds):
    maximum = (1 << width) - 1
    pairs = [(0, 0), (maximum, 1), (maximum, maximum), (1 << (width - 1), 1),
             ((1 << (width - 1)) - 1, maximum), (7, 7), (17, 0), (0, 7)]
    for seed in seeds:
        rng = random.Random(seed + width)
        pairs.extend((rng.randrange(1 << width), rng.randrange(1 << width)) for _ in range(4))
    return [{'a': a, 'b': b, 'stdin': (a.to_bytes(width // 8, 'little') + b.to_bytes(width // 8, 'little')).hex()}
            for a, b in pairs]


def prepare(out):
    out = external(out); out.mkdir(parents=True, exist_ok=False)
    seeds = [104729, 130363, 155921]
    programs = []
    for compiler in ('gcc', 'clang'):
        if not shutil.which(compiler): raise RuntimeError(f'{compiler} is required')
        for optimization in ('O0', 'O2'):
            for family in FAMILIES:
                name = f'{compiler}-{optimization}-{family}'
                item = build_program(out / name, family, compiler, optimization)
                item['id'] = name; item['cases'] = inputs(item['width'], seeds)
                programs.append(item)
    for index, spec in enumerate(({'exit': 0}, {'exit': 37}, {'exit': 255}, {'stdout': b'RGB.OK!\n'.hex()})):
        path = out / f'generation-{index}.elf'; path.write_bytes(compile_generation_target(spec))
        programs.append({'id': f'generation-{index}', 'family': 'existing_generation_curriculum',
                         'path': str(path), 'sha256': digest(path.read_bytes()), 'cases': [{'stdin': ''}]})
    compilers = {name: subprocess.check_output([name, '--version'], text=True).splitlines()[0] for name in ('gcc', 'clang')}
    protocol = {'spec': SPEC, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_hashes': source_hashes(), 'seeds': seeds, 'programs': programs,
                'compilers': compilers, 'python': platform.python_version(), 'torch': torch.__version__,
                'inference_budget': {'max_steps': 4096, 'trace_limit': 0},
                'stack_contract': 'C wrappers establish the same private ELF BSS stack in native and tensor runs; the initial OS stack is unused',
                'scope': 'authored static ELF differential conformance; no learned model or arbitrary-program claim',
                'gate': 'all completed supported cases equal native; unsupported/fault/budget remain failures; negative controls detected'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out / 'source').mkdir()
    for name in source_hashes():
        path = out / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return {'programs': len(programs), 'cases': sum(len(p['cases']) for p in programs),
            'protocol_sha256': digest((out / 'protocol.json').read_bytes())}


def evaluate(out):
    out = external(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['source_hashes'] != source_hashes(): raise ValueError('frozen implementation changed')
    for item in protocol['programs']:
        if digest(Path(item['path']).read_bytes()) != item['sha256']: raise ValueError('frozen executable changed')
    folder = out / 'evaluation'; folder.mkdir(exist_ok=False)
    native = ExecutableRunner(folder / 'native', cache=False)
    cores = {str(dtype): ExactX86Runner(dtype=dtype, **protocol['inference_budget'])
             for dtype in (torch.float32, torch.bfloat16)}
    statuses = Counter(); mismatches = []; histogram = Counter(); count = 0; instructions = 0
    start = time.perf_counter(); first_memory_disagreement = None
    with (folder / 'results.jsonl').open('x') as log:
        for program in protocol['programs']:
            raw = Path(program['path']).read_bytes()
            for index, inputs_ in enumerate(program['cases']):
                case = Case(stdin=bytes.fromhex(inputs_['stdin']))
                observed = {name: core.run(raw, case) for name, core in cores.items()}
                reference = native.run(raw, case)
                if reference['status'] != 'ok' or reference['returncode'] != 0 and program['family'] != 'existing_generation_curriculum':
                    raise RuntimeError(f'Native reference failed: {program["id"]} {reference}')
                for name, result in observed.items():
                    statuses[name + '/' + result['status']] += 1
                    equal = same_result(result, reference)
                    result['matches_native'] = equal
                    instructions += result['instructions']; histogram.update(result['histogram'])
                    if not equal:
                        mismatches.append({'program': program['id'], 'case': index, 'dtype': name,
                                           'status': result['status'], 'reason': result['reason'],
                                           'native': {k: reference[k] for k in ('status', 'returncode', 'stdout', 'stderr')},
                                           'actual': {k: result[k] for k in ('status', 'returncode', 'stdout', 'stderr')}})
                memories = [value['memory_sha256'] for value in observed.values()]
                if len(set(memories)) != 1 and first_memory_disagreement is None:
                    first_memory_disagreement = {'program': program['id'], 'case': index}
                record = {'program': program['id'], 'case': index, 'input': inputs_,
                          'compiled': observed, 'native': reference}
                log.write(json.dumps(record) + '\n'); log.flush(); count += 1
            print(json.dumps({'program': program['id'], 'cases_done': count,
                              'nonmatches': len(mismatches), 'elapsed_seconds': time.perf_counter() - start}), flush=True)
    controls = negative_controls(protocol, native)
    (folder / 'negative-controls.json').write_text(json.dumps(controls, indent=2))
    summary = {'protocol_sha256': digest((out / 'protocol.json').read_bytes()), 'source_hashes': source_hashes(),
               'programs': len(protocol['programs']), 'native_cases': count, 'tensor_cases': count * len(cores),
               'statuses': dict(statuses), 'nonmatches': mismatches, 'histogram': dict(histogram),
               'tensor_instructions': instructions, 'first_cross_dtype_memory_disagreement': first_memory_disagreement,
               'wall_seconds_diagnostic_only': time.perf_counter() - start,
               'native_calls': native.native_calls, 'reference_corrections': 0,
               'negative_controls': controls,
               'passed': not mismatches and first_memory_disagreement is None and controls['passed'],
               'scope': protocol['scope'], 'major_breakthrough_claim': False}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def negative_controls(protocol, native):
    program = next(item for item in protocol['programs'] if item['family'] == 'arithmetic64')
    raw = Path(program['path']).read_bytes()
    case = Case(stdin=bytes.fromhex(program['cases'][1]['stdin']))
    reference = native.run(raw, case)
    core = ExactX86Runner()
    baseline = core.run(raw, case)
    original = FixedWordAdder.forward

    def corrupt(module, a, b, carry_in=None):
        value, carry = original(module, a, b, carry_in)
        value = value.clone(); value[:, 0] = 1 - value[:, 0]
        return value, carry

    with patch.object(FixedWordAdder, 'forward', corrupt):
        corrupted = core.run(raw, case)
    rejected = core.run(compile_generation_target({'exit': 0}), Case(('unsupported-argv',)))
    budget = ExactX86Runner(max_steps=1).run(raw, case)
    checks = {'uncorrupted_matches': same_result(baseline, reference),
              'corruption_detected': not same_result(corrupted, reference),
              'unsupported_not_success': rejected['status'] == 'unsupported',
              'budget_not_success': budget['status'] == 'budget'}
    return {'passed': all(checks.values()), 'checks': checks,
            'corruption_status': corrupted['status'], 'corruption_reason': corrupted['reason'],
            'reference_corrections': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'evaluate'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    result = prepare(args.out) if args.phase == 'prepare' else evaluate(args.out)
    print(json.dumps(result, indent=2))
    if args.phase == 'evaluate' and not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
