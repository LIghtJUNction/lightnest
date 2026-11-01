"""Time the tensor ELF loader and x86 subset on CPU, GPU and CUDA Graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from lightnest.architecture.elf_tensor_loader import ELFTensorLoader
from lightnest.architecture.x86_internal import EXITED, X86Internal
from lightnest.training.corpus import external


class LoaderAndCPU(nn.Module):
    def __init__(self):
        super().__init__(); self.loader = ELFTensorLoader(); self.x86 = X86Internal()

    def forward(self, image, length):
        loaded = self.loader(image, length)
        result = self.x86(loaded['memory'], loaded['readable'], loaded['executable'], loaded['entry'], steps=16)
        return loaded['valid'], result['status'], result['exit_code'], result['output'], result['output_lengths']


def run(out: Path, image_path: Path, *, repeats=10):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if not 1 <= repeats <= 30: raise ValueError('invalid repeat count')
    raw = image_path.read_bytes()
    if not 64 <= len(raw) <= 768: raise ValueError('ELF must fit in 768 bytes')
    target = external(out); target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    timing = []
    for batch in (1, 32, 128):
        cpu_outputs = None
        for device in ('cpu', 'cuda'):
            model = LoaderAndCPU().to(device).eval()
            image = torch.zeros((batch,768), dtype=torch.long, device=device)
            image[:,:len(raw)] = torch.tensor(list(raw), device=device)
            lengths = torch.full((batch,),len(raw), dtype=torch.long, device=device)
            with torch.inference_mode():
                model(image, lengths)
                if device == 'cuda': torch.cuda.synchronize()
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter(); result = model(image, lengths)
                    if device == 'cuda': torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
                converted = [tensor.cpu() for tensor in result]
                if device == 'cpu': cpu_outputs = converted
                elif any(not torch.equal(a,b) for a,b in zip(cpu_outputs,converted)):
                    raise RuntimeError('CPU and CUDA outputs differ')
                if not bool(converted[0].all() and converted[1].eq(EXITED).all()):
                    raise RuntimeError('benchmark input did not pass and exit')
                timing.append({'batch': batch, 'device': device, 'seconds': samples,
                               'median_seconds': statistics.median(samples)})
                if device == 'cuda':
                    graph = torch.cuda.CUDAGraph(); static_image = image.clone(); static_length = lengths.clone()
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.cuda.graph(graph): graph_result = model(static_image, static_length)
                    torch.cuda.synchronize(); capture_seconds = time.perf_counter() - start
                    graph.replay(); torch.cuda.synchronize(); samples = []
                    for _ in range(repeats):
                        start = time.perf_counter(); graph.replay(); torch.cuda.synchronize()
                        samples.append(time.perf_counter() - start)
                    if any(not torch.equal(a,b.cpu()) for a,b in zip(cpu_outputs,graph_result)):
                        raise RuntimeError('CUDA graph output differs')
                    # Verify the captured graph reads the new image, not a cached result.
                    changed = raw.find(b'WZ\n')
                    if changed < 0: raise ValueError('benchmark expects the WZ example')
                    static_image[:,changed] = ord('X'); graph.replay(); torch.cuda.synchronize()
                    changed_result = graph_result[3][:,0,0].cpu()
                    if not bool(changed_result.eq(ord('X')).all()): raise RuntimeError('graph ignored modified input')
                    timing.append({'batch': batch, 'device': 'cuda_graph', 'seconds': samples,
                                   'median_seconds': statistics.median(samples), 'capture_seconds': capture_seconds,
                                   'changed_input_checked': True})
    report = {'image_sha256': hashlib.sha256(raw).hexdigest(), 'image_bytes': len(raw),
              'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'repeats': repeats, 'steps': 16, 'results': timing,
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                (Path(__file__), Path(__file__).parents[1] / 'architecture/x86_internal.py',
                                 Path(__file__).parents[1] / 'architecture/elf_tensor_loader.py')},
              'scope': 'ELF tensor loader and x86 rollout only; excludes neural proposal, input transfer and initial CUDA graph capture'}
    (target / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args(); print(json.dumps(run(args.out, args.image, repeats=args.repeats), indent=2))


if __name__ == '__main__': main()
