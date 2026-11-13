"""Frozen full-ELF CPU pilot; all data, logs and weights stay outside checkout.

Run: python -m lightnest.training.hnet_experiment --out /tmp/lightnest-hnet-run
This is a mechanism pilot, not the full locked-environment RGB benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import tempfile
import time

import torch
from torch.nn import functional as F

from lightnest.architecture.generate import ELFGenerator, condition_tensor, generation_loss
from lightnest.architecture.hnet import HNetELFGenerator, ratio_loss

BASE_COMMIT = '143e8900dede5c5364da4913507c4c4ed814fa09'
BASE_GENERATE_BLOB = 'c930f9ad685f2d6cfb2b7053e22bf45d0f042325'
VARIANTS = ('baseline', 'dense', 'fixed', 'dynamic')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def compile_target(spec):
    """Label construction only; same assembly/flags as corpus.py.

    Self-contained to avoid importing the corpus's isolation dependencies.
    The generation path never calls a compiler or inserts any target bytes.
    """
    if set(spec) == {'exit'}:
        assembly = f".global _start\n.text\n_start:\n mov ${spec['exit']},%edi\n mov $60,%eax\n syscall\n"
    else:
        payload = bytes.fromhex(spec['stdout'])
        assembly = ('.global _start\n.text\n_start:\n mov $1,%edi\n lea message(%rip),%rsi\n mov $8,%edx\n'
                    ' mov $1,%eax\n syscall\n xor %edi,%edi\n mov $60,%eax\n syscall\n.section .rodata\nmessage:\n.byte '
                    + ','.join(map(str, payload)) + '\n')
    assembly += '.section .note.GNU-stack,"",@progbits\n'
    with tempfile.TemporaryDirectory(prefix='hnet-label-') as temp:
        root = Path(temp); (root / 'target.s').write_text(assembly)
        subprocess.run(['as', '--64', '-o', str(root / 'target.o'), str(root / 'target.s')],
                       check=True, capture_output=True, timeout=10)
        subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(root / 'target'),
                        str(root / 'target.o')], check=True, capture_output=True, timeout=10)
        result = subprocess.run([str(root / 'target')], capture_output=True, timeout=2)
        expected_code = spec.get('exit', 0)
        expected_stdout = bytes.fromhex(spec.get('stdout', ''))
        if (result.returncode, result.stdout, result.stderr) != (expected_code, expected_stdout, b''):
            raise RuntimeError('reference program failed requested behavior')
        return (root / 'target').read_bytes()


def prepare(out):
    rows = []
    for status in range(256):
        split = 'test' if status % 8 == 4 else 'validation' if status % 8 == 0 else 'train'
        rows.append({'id': f'exit-{status:03d}', 'split': split, 'spec': {'exit': status}, 'family': 'exit'})
    for split, count, seed in (('train', 128, 11), ('validation', 16, 19), ('test', 32, 29)):
        rng = random.Random(seed)
        for i in range(count):
            payload = bytes(rng.choice(b'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*') for _ in range(8))
            rows.append({'id': f'stdout-{split}-{i:03d}', 'split': split,
                         'spec': {'stdout': payload.hex()}, 'family': 'stdout'})
    (out / 'objects').mkdir()
    for row in rows:
        raw = compile_target(row['spec'])
        row['sha256'], row['size'] = sha(raw), len(raw)
        (out / 'objects' / (row['sha256'] + '.elf')).write_bytes(raw)
    splits = {s: {r['sha256'] for r in rows if r['split'] == s} for s in ('train', 'validation', 'test')}
    if any(splits[a] & splits[b] for a, b in (('train', 'test'), ('train', 'validation'), ('test', 'validation'))):
        raise RuntimeError('exact-byte split overlap')
    dump(out / 'manifest.json', rows)
    return rows


def load_data(out, rows):
    return [dict(r, raw=(out / 'objects' / (r['sha256'] + '.elf')).read_bytes()) for r in rows]


def batch(rows):
    length = max(len(r['raw']) for r in rows) + 1
    previous = torch.zeros(len(rows), length, dtype=torch.long)
    targets = torch.zeros_like(previous); valid = torch.zeros_like(previous, dtype=torch.bool)
    for i, row in enumerate(rows):
        data = torch.tensor(list(row['raw']), dtype=torch.long); n = len(data)
        previous[i, 0] = 256; previous[i, 1:n + 1] = data
        targets[i, :n] = data; targets[i, n] = 256; valid[i, :n + 1] = True
    return previous, condition_tensor([r['spec'] for r in rows]), targets, valid


def make_model(variant, width, capacity):
    if variant == 'baseline': return ELFGenerator(width=width, capacity=capacity)
    return HNetELFGenerator(width=width, capacity=capacity, chunk_mode=variant, target_ratio=4)


@torch.inference_mode()
def teacher_metrics(model, rows, batch_size=16):
    model.eval(); total_bytes = bit_nll = byte_correct = loss_total = valid_total = 0
    boundary_total = padded_total = 0; per_case = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        x, c, y, valid = batch(subset); o = model(x, c, red_depth=2)
        loss_total += float(generation_loss(o, y, valid)) * len(subset)
        mask = valid & y.ne(256)
        expected = ((y[..., None] >> torch.arange(8)) & 1).float()
        nll = F.binary_cross_entropy_with_logits(o['bits'], expected, reduction='none').sum(-1) / math.log(2)
        predicted = ((o['bits'] >= 0).long() * (1 << torch.arange(8))).sum(-1)
        correct = predicted.eq(y)
        total_bytes += int(mask.sum()); bit_nll += float(nll[mask].sum())
        byte_correct += int((correct & mask).sum()); valid_total += int(valid.sum())
        boundary_total += int((o['boundaries'] & valid).sum()) if 'boundaries' in o else int(valid.sum())
        padded_total += int(o.get('padded_main_positions', valid.numel()))
        for i, row in enumerate(subset):
            n = len(row['raw']); variable = row['variable_positions']
            vc = sum(int(correct[i, j]) for j in variable)
            per_case.append({'id': row['id'], 'family': row['family'], 'target_sha256': row['sha256'],
                             'bytes': n, 'bpb': float(nll[i, :n].mean()),
                             'teacher_byte_correct': int(correct[i, :n].sum()),
                             'variable_byte_correct': vc, 'variable_byte_count': len(variable),
                             'teacher_exact': bool(correct[i, :n].all()),
                             'teacher_stop_correct': bool((o['stop'][i, :n] < 0).all() and o['stop'][i, n] >= 0)})
    return {'loss': loss_total / len(rows), 'bpb': bit_nll / total_bytes,
            'teacher_byte_accuracy': byte_correct / total_bytes,
            'boundary_fraction': boundary_total / valid_total,
            'bytes_per_chunk': valid_total / boundary_total,
            'padded_main_positions': padded_total, 'valid_positions': valid_total,
            'variable_byte_accuracy': sum(r['variable_byte_correct'] for r in per_case) /
                                      sum(r['variable_byte_count'] for r in per_case)}, per_case


@torch.inference_mode()
def generate_batch(model, rows):
    """Pure model rollout; no compiler, template, expected length or repair."""
    model.eval(); count = len(rows)
    c = condition_tensor([r['spec'] for r in rows])
    previous = torch.full((count, 1), 256, dtype=torch.long)
    hidden = None; stopped = torch.zeros(count, dtype=torch.bool)
    lengths = torch.full((count,), model.capacity, dtype=torch.long)
    output_bytes = torch.zeros(count, model.capacity, dtype=torch.uint8)
    start_time = time.perf_counter()
    for position in range(model.capacity + 1):
        o = model(previous, c, red_depth=2, hidden=hidden, position_offset=position)
        hidden = o['hidden']
        if not torch.isfinite(o['bits']).all() or not torch.isfinite(o['stop']).all():
            raise FloatingPointError('nonfinite autoregressive generation')
        stop_now = o['stop'][:, -1] >= 0
        lengths[stop_now & ~stopped] = position
        stopped = stopped | stop_now
        if bool(stopped.all()) or position == model.capacity: break
        value = ((o['bits'][:, -1] >= 0).long() * (1 << torch.arange(8))).sum(-1)
        output_bytes[:, position] = value.to(torch.uint8)
        previous = value[:, None]
    generated = [bytes(output_bytes[i, :int(lengths[i])].tolist()) for i in range(count)]
    return generated, stopped.tolist(), time.perf_counter() - start_time


def variable_positions(data):
    """Variable fields are derived from TRAIN bytes only, by family."""
    positions = {}
    for family in ('exit', 'stdout'):
        raw = [r['raw'] for r in data if r['split'] == 'train' and r['family'] == family]
        if len({len(x) for x in raw}) != 1: raise ValueError('fixed family size required in this pilot')
        positions[family] = [i for i in range(len(raw[0])) if len({r[i] for r in raw}) > 1]
    for r in data: r['variable_positions'] = positions[r['family']]
    return positions


def train_one(out, args, variant, seed, data):
    train = [r for r in data if r['split'] == 'train']
    val = [r for r in data if r['split'] == 'validation']
    torch.manual_seed(seed)
    model = make_model(variant, args.width, args.capacity)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed + 100000)
    schedule = torch.randint(len(train), (args.steps, args.batch_size), generator=generator)
    run_id = f'{variant}-s{seed}'; run = out / 'runs' / run_id; run.mkdir(parents=True)
    dump(run / 'batch_schedule.json', schedule.tolist())
    best_loss, best_step = math.inf, None
    train_seconds = 0.; logical_main = padded_main = positions_seen = 0
    with (run / 'train.jsonl').open('w') as log:
        for step, indices in enumerate(schedule, 1):
            x, c, y, valid = batch([train[i] for i in indices.tolist()])
            model.train(); start = time.perf_counter(); optim.zero_grad(set_to_none=True)
            o = model(x, c, red_depth=2); task_loss = generation_loss(o, y, valid)
            rloss = ratio_loss(o['boundary_probabilities'], o['boundaries'], 4, valid) if variant == 'dynamic' else task_loss * 0
            loss = task_loss + args.ratio_weight * rloss
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optim.step()
            train_seconds += time.perf_counter() - start
            positions_seen += int(valid.sum())
            logical_main += int((o['boundaries'] & valid).sum()) if 'boundaries' in o else int(valid.sum())
            padded_main += int(o.get('padded_main_positions', valid.numel()))
            record = {'step': step, 'task_loss': float(task_loss.detach()), 'ratio_loss': float(rloss.detach()),
                      'train_seconds': train_seconds,
                      'boundaries': int((o['boundaries'] & valid).sum()) if 'boundaries' in o else int(valid.sum()),
                      'valid_positions': int(valid.sum())}
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics, _ = teacher_metrics(model, val); record['validation'] = metrics
                if metrics['loss'] < best_loss:
                    best_loss, best_step = metrics['loss'], step
                    torch.save({'state_dict': model.state_dict(), 'variant': variant, 'seed': seed,
                                'width': args.width, 'capacity': args.capacity, 'step': step}, run / 'best.pt')
                print(json.dumps({'run': run_id, **record}), flush=True)
            log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
    result = {'run_id': run_id, 'variant': variant, 'seed': seed,
              'parameters': sum(p.numel() for p in model.parameters()), 'best_step': best_step,
              'best_validation_loss': best_loss, 'train_seconds': train_seconds,
              'logical_main_positions': logical_main, 'padded_main_positions': padded_main,
              'valid_positions_seen': positions_seen, 'schedule_sha256': sha(schedule.numpy().tobytes())}
    dump(run / 'training_summary.json', result)
    return result


def evaluate_one(out, training, data):
    run = out / 'runs' / training['run_id']
    ckpt = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model = make_model(ckpt['variant'], ckpt['width'], ckpt['capacity'])
    model.load_state_dict(ckpt['state_dict']); model.eval()
    test = [r for r in data if r['split'] == 'test']
    metrics, cases = teacher_metrics(model, test)
    generated, stopped, seconds = generate_batch(model, test)
    generated_dir = run / 'generated'; generated_dir.mkdir()
    for case, row, raw, did_stop in zip(cases, test, generated, stopped, strict=True):
        exact = raw == row['raw'] and did_stop
        # Never execute unknown outputs. Exact identity transfers the independently
        # checked behavior of the known, assembler-built reference.
        (generated_dir / (row['id'] + '.bin')).write_bytes(raw)
        case.update(generated_sha256=sha(raw), generated_bytes=len(raw), stopped=did_stop,
                    exact_generated=exact, elf_magic=raw.startswith(b'\x7fELF'),
                    verified_behavior_by_exact_identity=exact, unmatched_native_execution='not attempted',
                    free_variable_correct=sum(int(j < len(raw) and raw[j] == row['raw'][j]) for j in row['variable_positions']))
    with (run / 'test_cases.jsonl').open('w') as file:
        for case in cases: file.write(json.dumps(case) + '\n')
    metrics.update(exact_generated=sum(r['exact_generated'] for r in cases), test_count=len(cases),
                   generated_elf_magic=sum(r['elf_magic'] for r in cases), generation_seconds=seconds,
                   free_variable_accuracy=sum(r['free_variable_correct'] for r in cases) /
                                          sum(r['variable_byte_count'] for r in cases))
    for family in ('exit', 'stdout'):
        selected = [r for r in cases if r['family'] == family]
        metrics[family] = {'count': len(selected), 'exact': sum(r['exact_generated'] for r in selected),
                          'teacher_variable_accuracy': sum(r['variable_byte_correct'] for r in selected) /
                                                       sum(r['variable_byte_count'] for r in selected),
                          'free_variable_accuracy': sum(r['free_variable_correct'] for r in selected) /
                                                    sum(r['variable_byte_count'] for r in selected)}
    result = dict(training, test=metrics); dump(run / 'result.json', result)
    print('TEST ' + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--capacity', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--ratio-weight', type=float, default=.03)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]; out = args.out.resolve()
    if out == root or root in out.parents: parser.error('research outputs must be outside checkout')
    if min(args.steps, args.width, args.capacity, args.batch_size, args.eval_every, args.threads) < 1:
        parser.error('budgets and geometry must be positive')
    if not shutil.which('as') or not shutil.which('ld'): parser.error('GNU as and ld required')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = {'id': 'rgb-hnet-full-elf-pilot-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'base_commit': BASE_COMMIT, 'base_generate_git_blob': BASE_GENERATE_BLOB,
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'variants': VARIANTS, 'target_chunk_ratio': 4,
                'selection': 'minimum validation generation_loss; test used only after all training is complete',
                'budget_matching': 'same sampled examples, steps, width, optimizer; NOT FLOP or wall-time matched',
                'primary_metric': 'full-file free-generation byte exact match including stopping',
                'secondary_metrics': ['held-out BPB', 'teacher byte accuracy', 'variable byte accuracy', 'main chunk ratio', 'training wall time'],
                'scope': 'two assembler-built families, unseen exit values/eight-byte stdout; no arbitrary-program generalization',
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(),
                            'cuda': torch.cuda.is_available(), 'threads': args.threads},
                'tools': {t: subprocess.check_output([t, '--version']).decode().splitlines()[0] for t in ('as', 'ld')},
                'sources': {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / 'lightnest').rglob('*.py'))}}
    dump(out / 'protocol.json', protocol)
    rows = prepare(out); data = load_data(out, rows)
    if max(r['size'] for r in rows) > args.capacity: raise ValueError('capacity below target size')
    fields = variable_positions(data)
    dump(out / 'dataset_summary.json', {'manifest_sha256': sha((out / 'manifest.json').read_bytes()),
                                       'counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
                                       'sizes': sorted({r['size'] for r in rows}), 'train_variable_positions': fields,
                                       'exact_hash_overlap': 0, 'reference_native_checks': len(rows)})
    print('DATA ' + (out / 'dataset_summary.json').read_text(), flush=True)
    trained = []
    for i, seed in enumerate(args.seeds):
        order = VARIANTS[i % len(VARIANTS):] + VARIANTS[:i % len(VARIANTS)]
        for variant in order: trained.append(train_one(out, args, variant, seed, data))
    dump(out / 'training_complete.json', trained)
    results = [evaluate_one(out, training, data) for training in trained]
    summary = {}
    for variant in VARIANTS:
        subset = [r for r in results if r['variant'] == variant]
        summary[variant] = {'seeds': [r['seed'] for r in subset], 'parameters': subset[0]['parameters']}
        for metric in ('bpb', 'teacher_byte_accuracy', 'variable_byte_accuracy', 'free_variable_accuracy', 'exact_generated', 'bytes_per_chunk', 'generation_seconds'):
            values = [r['test'][metric] for r in subset]
            summary[variant][metric] = {'values': values, 'mean': statistics.mean(values),
                                        'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
        values = [r['train_seconds'] for r in subset]
        summary[variant]['train_seconds'] = {'values': values, 'mean': statistics.mean(values),
                                            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
    dump(out / 'results.json', results); dump(out / 'summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
