"""Fit a limited internal openat model to controlled host filesystem observations.

The observed files are experiment fixtures, not an OS image. The model is not
an implementation of a filesystem or a general Linux environment.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

from lightnest.architecture.linux_environment import LearnedOpenat
from lightnest.architecture.x86_internal import EXITED, X86Internal
from lightnest.runtime.internal_elf import load_static_elf
from lightnest.training.corpus import external


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode(paths, size=128):
    rows = []
    for path in paths:
        data = os.fsencode(path) + b'\0'
        if len(data) > size:
            raise ValueError('fixture path exceeds learned environment capacity')
        rows.append(list(data) + [0] * (size - len(data)))
    return torch.tensor(rows, dtype=torch.long)


def _build_elf(folder: Path, path: Path, flags: int, index: int) -> bytes:
    source = ('.global _start\n.text\n_start:\nmov $257,%eax\nmov $-100,%edi\nlea path(%rip),%rsi\n'
              f'mov ${flags},%edx\nsyscall\nmov %eax,%edi\nmov $60,%eax\nsyscall\n'
              f'.section .rodata\npath: .asciz "{path}"\n.section .note.GNU-stack,"",@progbits\n')
    asm, obj, exe = (folder / f'case-{index}{suffix}' for suffix in ('.s', '.o', '.elf'))
    asm.write_text(source)
    subprocess.run(['as', '--64', '-o', str(obj), str(asm)], check=True, capture_output=True)
    subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(exe), str(obj)], check=True, capture_output=True)
    return exe.read_bytes()


def run(out: Path, *, device='cuda', steps=500):
    if not 1 <= steps <= 10000: raise ValueError('invalid training steps')
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    root = external(out); root.mkdir(parents=True, exist_ok=False)
    fixture = root / 'observed-files'; fixture.mkdir()
    protocol = {'scope': 'openat outcomes for controlled local fixture paths, not a Linux implementation',
                'split': {'train': [0, 39], 'validation': [40, 49], 'test': [50, 59]},
                'path_classes': ['writable', 'readonly', 'missing'], 'flags': [0, 1],
                'classes': {'0': 'success normalized to fd 3', '1': 'ENOENT', '2': 'EACCES'},
                'steps': steps, 'seed': 20261101, 'device': device,
                'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'environment_source_sha256': _hash(Path(__file__).resolve().parents[1] / 'architecture/linux_environment.py'),
                'cpu_source_sha256': _hash(Path(__file__).resolve().parents[1] / 'architecture/x86_internal.py'),
                'trainer_source_sha256': _hash(Path(__file__)),
                'selection': 'lowest validation cross entropy every 50 updates, then test once'}
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    records = []
    for i in range(60):
        for category in ('writable', 'readonly', 'missing'):
            path = fixture / f'{category}-{i:03}.txt'
            if category != 'missing':
                path.write_bytes(b'fixture\n')
                path.chmod(0o600 if category == 'writable' else 0o444)
            for flags in (0, 1):
                try:
                    fd = os.open(path, flags | os.O_CLOEXEC); os.close(fd); label = 0
                except OSError as error:
                    if error.errno not in (errno.ENOENT, errno.EACCES): raise
                    label = 1 if error.errno == errno.ENOENT else 2
                expected = 1 if category == 'missing' else 2 if category == 'readonly' and flags == 1 else 0
                if label != expected: raise RuntimeError('host fixture permission behavior differs from protocol')
                records.append({'path': str(path), 'flags': flags, 'label': label,
                                'split': 'train' if i < 40 else 'validation' if i < 50 else 'test'})
    (root / 'observations.json').write_text(json.dumps(records, indent=2) + '\n')
    train = [row for row in records if row['split'] == 'train']
    validation = [row for row in records if row['split'] == 'validation']
    def tensors(rows):
        return (_encode([row['path'] for row in rows]).to(device),
                torch.tensor([row['flags'] for row in rows], dtype=torch.long, device=device),
                torch.tensor([row['label'] for row in rows], dtype=torch.long, device=device))
    torch.manual_seed(20261101); torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    model = LearnedOpenat(path_size=128, width=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    train_x, train_f, train_y = tensors(train); val_x, val_f, val_y = tensors(validation)
    best, best_state, best_step = float('inf'), None, 0
    if device == 'cuda': torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x, train_f), train_y); loss.backward(); optimizer.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad(): val_loss = float(F.cross_entropy(model(val_x, val_f), val_y))
            if val_loss < best:
                best, best_step = val_loss, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if device == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state); model.eval()
    torch.save({'state_dict': best_state, 'config': {'path_size': 128, 'width': 8},
                'protocol': protocol, 'best_step': best_step}, root / 'environment.pt')
    # Held-out labels are first used after the validation checkpoint is selected.
    test = [row for row in records if row['split'] == 'test']
    test_x, test_f, test_y = tensors(test)
    with torch.inference_mode(): predicted = model(test_x, test_f).argmax(dim=1)
    correct = int(predicted.eq(test_y).sum())
    counts = torch.bincount(test_y, minlength=3).cpu().tolist()
    # Independent validation inputs are actual x86-64 ELF bytes, assembled only
    # to construct test files. Rollouts themselves use only tensor operations.
    import tempfile
    integrated = []
    with tempfile.TemporaryDirectory(prefix='lightnest-openat-test-') as temp:
        machine = X86Internal(environment=model).to(device).eval()
        for index, row in enumerate(test[:6]):
            raw = _build_elf(Path(temp), Path(row['path']), row['flags'], index)
            loaded = load_static_elf(raw, device=device)
            with torch.inference_mode():
                result = machine(loaded['memory'], loaded['readable'], loaded['executable'], loaded['entry'], steps=12)
            expected = (3, 254, 243)[row['label']]
            integrated.append({'image_sha256': hashlib.sha256(raw).hexdigest(), 'path': row['path'],
                               'flags': row['flags'], 'expected_exit': expected, 'actual_exit': int(result['exit_code'][0]),
                               'status': int(result['status'][0]),
                               'correct': int(result['status'][0]) == EXITED and int(result['exit_code'][0]) == expected})
    result = {'best_step': best_step, 'validation_loss': best, 'training_seconds': elapsed,
              'heldout': {'correct': correct, 'total': len(test), 'majority_baseline': max(counts), 'class_counts': counts},
              'x86_elf_integration': {'correct': sum(row['correct'] for row in integrated), 'total': len(integrated), 'cases': integrated},
              'caveat': 'the path names encode fixture class; no arbitrary filesystem or general OS claim'}
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(args.out, device=args.device, steps=args.steps), indent=2))


if __name__ == '__main__': main()
