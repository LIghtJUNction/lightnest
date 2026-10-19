"""Build real Lean proofs, audit axioms, and check live Python conformance.

No weights, training data, GPU, or external model API. Missing Lean is a failure,
not a skipped test. Generated proof cases and build files stay in a temporary
workspace. Proof-source review is still required; this is not a code sandbox.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

try:
    from .formal_contracts import aligned_comparisons, retention_gate, schedule_cost
except ImportError:
    from formal_contracts import aligned_comparisons, retention_gate, schedule_cost

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = 'leanprover/lean4:v4.35.0'
FORBIDDEN = re.compile(r'\b(sorry|sorryAx|admit|axiom|native_decide|unsafe|partial|extern|implemented_by)\b')


def without_comments_and_strings(text: str) -> str:
    """Preserve line breaks; support nested Lean block comments."""
    result = []
    i = depth = 0
    string = False
    while i < len(text):
        pair = text[i:i+2]
        if depth:
            if pair == '/-':
                depth += 1; result.extend('  '); i += 2; continue
            if pair == '-/':
                depth -= 1; result.extend('  '); i += 2; continue
            result.append('\n' if text[i] == '\n' else ' '); i += 1; continue
        if string:
            if text[i] == '\\' and i+1 < len(text):
                result.extend('  '); i += 2; continue
            if text[i] == '"':
                string = False
            result.append('\n' if text[i] == '\n' else ' '); i += 1; continue
        if pair == '/-':
            depth = 1; result.extend('  '); i += 2; continue
        if pair == '--':
            while i < len(text) and text[i] != '\n':
                result.append(' '); i += 1
            continue
        if text[i] == '"':
            string = True; result.append(' '); i += 1; continue
        result.append(text[i]); i += 1
    if depth or string:
        raise ValueError('unterminated Lean comment or string')
    return ''.join(result)


def check_sources(formal: Path) -> int:
    if (formal / 'lean-toolchain').read_text().strip() != TOOLCHAIN:
        raise ValueError('unexpected Lean toolchain; update the pin and validation together')
    root_text = (formal / 'LightNest.lean').read_text()
    modules = sorted((formal / 'LightNest').rglob('*.lean'))
    expected = {'LightNest.' + '.'.join(p.relative_to(formal / 'LightNest').with_suffix('').parts)
                for p in modules}
    imported = set(re.findall(r'^import\s+(\S+)\s*$', root_text, re.M))
    if not modules or imported != expected:
        raise ValueError('root imports must cover every proof module exactly')
    count = 0
    for path in modules + [formal / 'LightNest.lean']:
        source = without_comments_and_strings(path.read_text())
        if match := FORBIDDEN.search(source):
            raise ValueError(f'{path.name}: forbidden proof escape: {match[0]}')
        if re.search(r'\brun_cmd\b|\brun_tac\b|#eval|#reduce', source):
            raise ValueError(f'{path.name}: executable elaboration belongs outside proof modules')
        for imp in re.findall(r'^import\s+(\S+)\s*$', source, re.M):
            if imp != 'Std' and imp not in expected:
                raise ValueError(f'{path.name}: unreviewed import {imp}')
        count += len(re.findall(r'^theorem\s+', source, re.M))
    if count < 20:
        raise ValueError('proof inventory unexpectedly small')
    return count


def load_guard(root: Path):
    path = root / 'baseline/LightNest_RGB_v2/rgb/guard.py'
    spec = importlib.util.spec_from_file_location('_lightnest_conformance_guard', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load current patch guard')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_depths(root: Path) -> tuple[int, ...]:
    source = ast.parse((root / 'hybrid/model.py').read_text())
    for node in source.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEPTHS' for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError('cannot find literal Hybrid DEPTHS; update the contract adapter')


def lean_bool(value: bool) -> str:
    return 'true' if value else 'false'


def conformance_source(root: Path) -> tuple[str, dict[str, int]]:
    guard = load_guard(root)
    rng = random.Random(20261020)
    lines = ['import Lean', 'import LightNest', 'set_option maxRecDepth 16384',
             'set_option maxHeartbeats 4000000', 'namespace LightNest.Conformance']
    counts = {'patch': 0, 'schedule': 0, 'retention': 0}

    def theorem(category: str, statement: str):
        number = counts[category]
        lines.append(f'theorem {category}_{number} : {statement} := by decide')
        counts[category] += 1

    def patch_case(code: bytes, edits: list, limit: int = 8, base_ok: bool = True):
        before = bytes(code)
        digest = hashlib.sha256(code).hexdigest() if base_ok else '0' * 64
        try:
            result = guard.candidate(code, digest, edits, max_edits=limit)
            expected = f'some {list(result)}'
        except ValueError:
            expected = 'none'
        if code != before:
            raise AssertionError('runtime mutated the original input')
        lean_edits = '[' + ', '.join(f'⟨{e.offset}, {e.before}, {e.after}⟩' for e in edits) + ']'
        theorem('patch', f'Patch.checkedPatch {list(code)} {lean_edits} {limit} {lean_bool(base_ok)} = {expected}')

    for _ in range(48):
        code = rng.randbytes(4 * rng.randint(1, 16))
        positions = rng.sample(range(len(code)), rng.randint(0, min(8, len(code))))
        edits = [guard.Edit(i, code[i], code[i] ^ rng.randint(1, 255)) for i in positions]
        patch_case(code, edits)
        patch_case(code, edits, base_ok=False)
        if edits:
            e = edits[0]
            patch_case(code, [guard.Edit(e.offset, e.before ^ 1, e.after)])
            patch_case(code, [e, e])
            patch_case(code, [guard.Edit(e.offset, e.before, e.before)])
        patch_case(code, [guard.Edit(len(code), 0, 1)])
        patch_case(code, [guard.Edit(0, code[0], 256)])
    for size in (0, 1, 3, 4, 7, 1024, 1028):
        patch_case(bytes(size), [])
    patch_case(bytes(12), [guard.Edit(i, 0, 1) for i in range(9)])
    patch_case(bytes(4), [], limit=9)
    patch_case(bytes(4), [guard.Edit(0, 0, 1)], limit=0)
    patch_case(bytes(4), [], limit=0)

    schedules = [checkpoint_depths(root), (), (64,), (4, 4, 64), (8, 4, 64),
                 (4, 8, 32), (0, 64), (4, 65), (4, 8, 16, 32, 64)]
    for _ in range(32):
        schedules.append(tuple(sorted(rng.sample(range(1, 64), rng.randint(0, 8)))) + (64,))
    for depths in schedules:
        try:
            expected = f'some {schedule_cost(depths)}'
        except ValueError:
            expected = 'none'
        theorem('schedule', f'Compute.scheduleCost 0 64 {list(depths)} = {expected}')

    # Every valid Boolean row and random aligned populations. Invalid envelopes
    # (duplicate IDs, changed inputs, non-booleans) are covered by Python tests.
    combinations = [(p, c, f) for p, c, f in itertools.product((False, True), repeat=3)
                    if not (c and f)]
    populations = [[row] for row in combinations]
    populations += [[rng.choice(combinations) for _ in range(rng.randint(1, 16))] for _ in range(48)]
    for rows in populations:
        parent = [{'id':str(i), 'initial_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
                   'strict':p, 'false_edit':False} for i, (p, _, _) in enumerate(rows)]
        child = [p | {'strict':c, 'false_edit':f} for p, (_, c, f) in zip(parent, rows, strict=True)]
        expected = retention_gate(parent, list(reversed(child)))['retention_passed']
        aligned = aligned_comparisons(parent, child)
        values = '[' + ', '.join('⟨' + ', '.join(map(lean_bool, r)) + '⟩' for r in aligned) + ']'
        theorem('retention', f'Promotion.check {values} = {lean_bool(expected)}')
    theorem('retention', 'Promotion.check [] = false')
    lines.append('end LightNest.Conformance')
    audit = (root / 'formal/Audit.lean').read_text()
    lines.extend(line for line in audit.splitlines() if not line.startswith('import '))
    return '\n'.join(lines) + '\n', counts


def command(args: list[str], cwd: Path, *, reject: str | None = None) -> str:
    started = time.monotonic()
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=600)
    print('$ ' + ' '.join(args), flush=True)
    print(result.stdout, flush=True)
    if reject is not None:
        if result.returncode == 0 or reject not in result.stdout:
            raise RuntimeError(f'negative control did not fail for the expected reason: {reject}')
    elif result.returncode:
        if os.environ.get('GITHUB_ACTIONS'):
            message = result.stdout[-6000:].replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
            print(f'::error::{message}', flush=True)
        raise RuntimeError(f'command failed ({result.returncode})')
    print(f'Elapsed: {time.monotonic() - started:.3f}s', flush=True)
    return result.stdout


def verify(root: Path = ROOT) -> dict:
    formal = root / 'formal'
    stated = check_sources(formal)
    if shutil.which('lake') is None:
        raise RuntimeError('Lake is missing. Install the pinned Lean toolchain. No Lean proof was checked.')
    generated, counts = conformance_source(root)
    with tempfile.TemporaryDirectory(prefix='lightnest-lean-') as temporary:
        stage = Path(temporary) / 'formal'
        shutil.copytree(formal, stage, ignore=shutil.ignore_patterns('.lake'))
        version = command(['lake', 'env', 'lean', '--version'], stage)
        if not re.search(r'\b4\.35\.0\b', version):
            raise RuntimeError('wrong Lean compiler version')
        command(['lake', 'build'], stage)
        audit_output = command(['lake', 'env', 'lean', '-DwarningAsError=true', 'Audit.lean'], stage)
        command(['lake', 'env', 'leanchecker', 'LightNest'], stage)
        (stage / 'Conformance.lean').write_text(generated)
        command(['lake', 'env', 'lean', '-DwarningAsError=true', 'Conformance.lean'], stage)
        audit = '\n'.join(line for line in (stage / 'Audit.lean').read_text().splitlines()
                          if not line.startswith('import '))
        negatives = [
            ('false_theorem', 'theorem LightNest.bad : (1 : Nat) = 2 := by decide', 'decide'),
            ('placeholder', 'theorem LightNest.missing : True := by sorry', 'sorry'),
            ('custom_axiom', 'axiom LightNest.fake : False\ntheorem LightNest.bad : False := LightNest.fake\n' + audit, 'unapproved'),
            ('native_proof', 'theorem LightNest.native : (2 : Nat) + 2 = 4 := by native_decide\n' + audit, 'unapproved'),
            ('unsafe_early_exit', 'theorem LightNest.bad_exit : LightNest.Compute.delayedOutput 2 = LightNest.Compute.delayedOutput 3 := by decide', 'decide'),
        ]
        for name, code, reason in negatives:
            path = stage / f'Negative_{name}.lean'
            path.write_text('import Lean\nimport LightNest\n' + code + '\n')
            command(['lake', 'env', 'lean', '-DwarningAsError=true', path.name], stage, reject=reason)
    return {
        'passed': True, 'toolchain': TOOLCHAIN, 'stated_theorems': stated,
        'axiom_audit': audit_output.strip(), 'kernel_recheck': True,
        'conformance_cases': counts, 'negative_controls_rejected': len(negatives),
        'model_weights_changed': False, 'new_accuracy_measurement': False,
        'scope': 'Lean specifications plus finite Python agreement; not complete Python/PyTorch verification.',
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(formal.rglob('*')) if p.is_file() and '.lake' not in p.parts},
        'runtime_guard_sha256': hashlib.sha256((root / 'baseline/LightNest_RGB_v2/rgb/guard.py').read_bytes()).hexdigest(),
        'hybrid_schedule_source_sha256': hashlib.sha256((root / 'hybrid/model.py').read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='optional NEW receipt outside the repository')
    args = parser.parse_args()
    if args.report and (args.report.exists() or args.report.resolve().is_relative_to(ROOT)):
        raise ValueError('use a new receipt path outside the source repository')
    report = verify()
    text = json.dumps(report, indent=2) + '\n'
    print(text)
    if args.report:
        with args.report.open('x', encoding='utf-8') as output:
            output.write(text)


if __name__ == '__main__':
    main()
