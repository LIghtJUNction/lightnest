"""Train a tiny full-ELF proposer and learned openat model on CUDA.

The task uses two-character outputs from one assembler-built ELF skeleton.
Held-out pairs test composition, not new algorithms or natural language.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from torch.nn import functional as F

from lightnest.architecture.linux_environment import LearnedOpenat
from lightnest.architecture.x86_world import X86World
from lightnest.runtime.execute import Case, ExecutableRunner
from lightnest.training.corpus import external


def make_elf(root: Path, first: int, second: int) -> bytes:
    source = ('.global _start\n.text\n_start:\nmov $1,%edi\nlea msg(%rip),%rsi\nmov $3,%edx\n'
              'mov $1,%eax\nsyscall\nxor %edi,%edi\nmov $60,%eax\nsyscall\n'
              f'.section .rodata\nmsg: .byte {first},{second},10\n.section .note.GNU-stack,"",@progbits\n')
    asm, obj, exe = root / 'fixture.s', root / 'fixture.o', root / 'fixture.elf'
    asm.write_text(source)
    subprocess.run(['as', '--64', '-o', str(obj), str(asm)], check=True, capture_output=True)
    subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(exe), str(obj)], check=True, capture_output=True)
    return exe.read_bytes()


def encode_paths(paths, width=128):
    rows = []
    for path in paths:
        value = os.fsencode(path) + b'\0'
        if len(value) > width: raise ValueError('path too long')
        rows.append(list(value) + [0] * (width - len(value)))
    return torch.tensor(rows, dtype=torch.long)


def run(out: Path, *, environment_checkpoint: Path, observations: Path, device='cuda', steps=500, alphabet='ABCD'):
    alphabet = alphabet.encode('ascii')
    if len(alphabet) != 4 or len(set(alphabet)) != 4 or any(not 32 <= c < 127 for c in alphabet):
        raise ValueError('alphabet must contain four distinct printable ASCII characters')
    if not 1 <= steps <= 10000: raise ValueError('invalid steps')
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    root = external(out); root.mkdir(parents=True, exist_ok=False)
    data_dir = root / 'dataset'; data_dir.mkdir()
    rows = []
    with tempfile.TemporaryDirectory(prefix='lightnest-world-label-') as tmp:
        for a in range(4):
            for b in range(4):
                raw = make_elf(Path(tmp), alphabet[a], alphabet[b])
                split_id = (a + b) % 4
                split = 'train' if split_id in (0, 1) else 'validation' if split_id == 2 else 'test'
                name = f'{a}-{b}.elf'; (data_dir / name).write_bytes(raw)
                rows.append({'a': a, 'b': b, 'split': split, 'file': name, 'size': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest()})
    if any(row['size'] > 768 for row in rows): raise ValueError('fixture exceeds capacity')
    observed = json.loads(Path(observations).read_text())
    os_train = [r for r in observed if r['split'] == 'train']
    os_test = [r for r in observed if r['split'] == 'test']
    checkpoint = torch.load(environment_checkpoint, map_location='cpu', weights_only=True)
    protocol = {'scope': 'two-character output, one shared ELF skeleton; no natural language or unseen algorithm',
                'splits': 'train if (a+b)%4 in {0,1}, validation 2, test 3', 'alphabet': alphabet.decode(),
                'rows': rows, 'steps': steps, 'seed': 20261102, 'device': device,
                'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                'world_source_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'architecture/x86_world.py').read_bytes()).hexdigest(),
                'loader_source_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'architecture/elf_tensor_loader.py').read_bytes()).hexdigest(),
                'cpu_source_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'architecture/x86_internal.py').read_bytes()).hexdigest(),
                'trainer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'environment_parent_sha256': hashlib.sha256(Path(environment_checkpoint).read_bytes()).hexdigest(),
                'observations_sha256': hashlib.sha256(Path(observations).read_bytes()).hexdigest(),
                'selection': 'lowest validation ELF byte loss every 50 steps, then held-out test once',
                'training': 'supervised byte and length labels plus separate openat classification loss; no gradient through hard execution'}
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.manual_seed(20261102); torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    env = LearnedOpenat(**checkpoint['config']); env.load_state_dict(checkpoint['state_dict'])
    model = X86World(condition_width=8, candidates=1, image_capacity=768, environment=env).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    def batch(items):
        condition = torch.zeros((len(items), 8))
        target = torch.zeros((len(items), 768), dtype=torch.long)
        length = torch.empty(len(items), dtype=torch.long)
        for i, row in enumerate(items):
            condition[i,row['a']] = 1; condition[i,4 + row['b']] = 1
            raw = (data_dir / row['file']).read_bytes()
            target[i,:len(raw)] = torch.tensor(list(raw)); length[i] = len(raw)-1
        return condition.to(device), target.to(device), length.to(device)
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'validation']
    tx, ty, tl = batch(train); vx, vy, vl = batch(val)
    ep = encode_paths([r['path'] for r in os_train], checkpoint['config']['path_size']).to(device)
    ef = torch.tensor([r['flags'] for r in os_train], device=device)
    ey = torch.tensor([r['label'] for r in os_train], device=device)
    def byte_loss(x, y, length):
        logits = model.proposal(x).reshape(len(x), 768, 256)
        length_logits = model.length(x).reshape(len(x), 768)
        return F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + F.cross_entropy(length_logits, length)
    best, best_step, best_state = float('inf'), 0, None
    if device == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = byte_loss(tx, ty, tl) + F.cross_entropy(model.x86.environment(ep, ef), ey)
        loss.backward(); optimizer.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad(): val_loss = float(byte_loss(vx, vy, vl))
            if val_loss < best:
                best, best_step = val_loss, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if device == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state); model.eval()
    torch.save({'state_dict': best_state, 'protocol': protocol, 'best_step': best_step}, root / 'world.pt')
    # Test queries begin only after the checkpoint is selected.
    test = [r for r in rows if r['split'] == 'test']
    cx, truth, truth_lengths = batch(test)
    expected = torch.zeros((len(test), 64), dtype=torch.long, device=device)
    for i, row in enumerate(test): expected[i,:3] = torch.tensor([alphabet[row['a']], alphabet[row['b']], 10], device=device)
    with torch.inference_mode():
        world = model(cx, expected, torch.full((len(test),),3,device=device),
                      torch.zeros(len(test),dtype=torch.long,device=device), steps=16, rounds=2)
    otx = encode_paths([r['path'] for r in os_test], checkpoint['config']['path_size']).to(device)
    otf = torch.tensor([r['flags'] for r in os_test],device=device)
    oty = torch.tensor([r['label'] for r in os_test],device=device)
    with torch.inference_mode(): os_correct = int(model.x86.environment(otx, otf).argmax(1).eq(oty).sum())
    runner = ExecutableRunner(root / 'native-final', cache=False)
    cases = []
    for i, row in enumerate(test):
        accepted = bool(world['accepted'][i]); raw = bytes(world['image'][i,:int(world['length'][i])].tolist())
        actual = runner.run(raw, Case()) if accepted else None
        desired = bytes([alphabet[row['a']], alphabet[row['b']], 10])
        native_ok = bool(actual and actual['status'] == 'ok' and actual['returncode'] == 0
                         and bytes.fromhex(actual['stdout']) == desired and actual['stderr'] == '')
        cases.append({'pair': desired[:2].decode(), 'accepted_internal': accepted,
                      'round_passed': world['round_passed'][i].tolist(), 'bytes': len(raw),
                      'image_sha256': hashlib.sha256(raw).hexdigest(),
                      'exact_reference': raw == (data_dir / row['file']).read_bytes(), 'native_correct': native_ok})
    result = {'best_step': best_step, 'validation_loss': best, 'training_seconds': elapsed,
              'heldout': {'internal_accepted': sum(r['accepted_internal'] for r in cases),
                          'native_correct': sum(r['native_correct'] for r in cases),
                          'exact_reference': sum(r['exact_reference'] for r in cases), 'total': len(cases), 'cases': cases},
              'heldout_openat': {'correct': os_correct, 'total': len(os_test)},
              'limits': 'one known skeleton; separate supervised environment loss; no natural language or unseen algorithm'}
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--environment-checkpoint', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--alphabet', default='ABCD')
    args = parser.parse_args()
    print(json.dumps(run(args.out, environment_checkpoint=args.environment_checkpoint,
                         observations=args.observations, device=args.device, steps=args.steps, alphabet=args.alphabet), indent=2))


if __name__ == '__main__': main()
