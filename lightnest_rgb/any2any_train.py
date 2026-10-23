"""CPU Any2Any byte-stream pretraining on mixed local artifacts."""
from __future__ import annotations

import json
import gzip
import io
import sqlite3
import zipfile
import ast
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .any2any import ArtifactCore, ArtifactRequest, KINDS, kind_index
from .artifact_types import detect_kind


def collect_files():
    paths=[]
    for root in (Path('/usr/bin'), Path('/usr/lib'), Path('/usr/lib/python3.13/site-packages'), Path('/etc'), Path('/usr/share')):
        if root.exists():
            found=0
            for path in root.rglob('*'):
                if path.is_file() and path.stat().st_size <= 65536:
                    paths.append(path); found += 1
                if found >= 256: break
    return paths


def code_summary(raw: bytes) -> list[float]:
    """Generic Python structure labels, computed only for training samples."""
    try:
        tree=ast.parse(raw.decode('utf-8')); nodes=list(ast.walk(tree))
        return [min(len(raw)/65536,1.), min(raw.count(b'\n')/2048,1.), min(sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes)/64,1.), min(sum(isinstance(n,ast.ClassDef) for n in nodes)/32,1.), min(sum(isinstance(n,ast.Import) or isinstance(n,ast.ImportFrom) for n in nodes)/64,1.)]
    except (SyntaxError,UnicodeDecodeError): return [0.,0.,0.,0.,0.]


def train_any2any(out, *, steps=256, seed=104734, width=64):
    out=Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    paths=collect_files(); samples=[]
    for path in paths:
        try:
            raw=path.read_bytes(); kind=detect_kind(raw,path.name)
            if raw: samples.append((path.name,raw,kind))
        except OSError: pass
    # Add real-content containers so one run spans structured, compressed and
    # archive formats without copying a fixed template into the model.
    if samples:
        base = samples[:32]
        for index, (_, raw, _) in enumerate(base):
            samples.append((f"sample-{index}.json", json.dumps({"index": index, "sha256": __import__('hashlib').sha256(raw).hexdigest(), "bytes": list(raw[:64])}).encode(), "json"))
            samples.append((f"sample-{index}.gz", gzip.compress(raw[:8192]), "compressed"))
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive: archive.writestr("payload.bin", raw[:8192])
            samples.append((f"sample-{index}.zip", stream.getvalue(), "archive"))
    if not samples: raise RuntimeError('no mixed artifact samples')
    model=ArtifactCore(width=width,heads=4,max_bytes=65536); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng=random.Random(seed); started=time.perf_counter(); losses=[]
    with (out/'updates.jsonl').open('x') as log:
        for step in range(steps):
            name,raw,kind=rng.choice(samples); length=min(len(raw),256); start=0 if len(raw)<=length else rng.randrange(len(raw)-length)
            window=raw[start:start+length]; request=ArtifactRequest(kind,'understand and transform',name)
            encoded,_=model.encode(window,request); abstract,guidance=model.think(encoded,depth=4)
            heads=model.heads(encoded,guidance[:,None]); output=heads['red_logits']
            target=torch.tensor(list(window),device=output.device)[None]
            kind_target=torch.tensor([kind_index(kind)],device=output.device)
            summary_target=torch.tensor([code_summary(raw)],device=output.device)
            loss=F.cross_entropy(output.reshape(-1,257),target.reshape(-1)) + .25 * F.cross_entropy(heads['kind_logits'],kind_target) + .5 * F.mse_loss(heads['summary'],summary_target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
            if (step+1)%32==0:
                event={'step':step+1,'loss':sum(losses)/len(losses),'samples':len(samples),'wall_seconds':time.perf_counter()-started}
                log.write(json.dumps(event)+'\n'); log.flush(); print(event,flush=True); losses=[]
    torch.save({'architecture':'lightnest-any2any-v1','kind':'artifact-core','config':{'width':width,'heads':4,'max_bytes':65536},'state_dict':model.state_dict(),'metadata':{'seed':seed,'steps':steps,'samples':len(samples),'kinds':sorted({k for _,_,k in samples}),'status':'research_candidate'}},out/f'any2any-{seed}.pt')
    (out/'manifest.json').write_text(json.dumps({'samples':len(samples),'kinds':sorted({k for _,_,k in samples}),'steps':steps})+'\n')
