"""Compare the migrated architectures against the pinned pre-migration Git source."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from lightnest.architecture.weightlang_io import save_model
from weightlang.tensor_program import load_tensor_artifact

ROOT = Path(__file__).resolve().parents[1]


def tensors(value, path="output"):
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, dict):
        return {k: v for name, item in value.items() for k, v in tensors(item, f"{path}.{name}").items()}
    if isinstance(value, (list, tuple)):
        return {k: v for i, item in enumerate(value) for k, v in tensors(item, f"{path}.{i}").items()}
    if is_dataclass(value):
        return tensors({f.name: getattr(value, f.name) for f in fields(value)}, path)
    return {}


def compare(a, b):
    left, right = tensors(a), tensors(b)
    assert left.keys() == right.keys(), (left.keys(), right.keys())
    maximum = 0.
    for key, x in left.items():
        y = right[key]
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.is_floating_point():
            assert torch.isfinite(x).all() and torch.isfinite(y).all(), key
            maximum = max(maximum, float((x-y).abs().max().detach()) if x.numel() else 0.)
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5, msg=key)
        else:
            assert torch.equal(x, y), key
    return maximum, len(left)


def check_case(name, old, new, call, kwargs, output):
    # Explicit identity mapping: names, dtypes and shapes must all match.
    source = old.state_dict(); expected = new.state_dict()
    assert source.keys() == expected.keys()
    for key in source:
        assert source[key].shape == expected[key].shape and source[key].dtype == expected[key].dtype, key
    new.load_state_dict(source, strict=True)
    old.eval(); new.eval()
    with torch.no_grad():
        error, count = compare(call(old), call(new))
    old.train(); new.train()
    torch.manual_seed(92); before = call(old)
    torch.manual_seed(92); after = call(new)
    train_error, _ = compare(before, after)
    loss_a = sum(t.float().square().mean() for t in tensors(before).values() if t.requires_grad)
    loss_b = sum(t.float().square().mean() for t in tensors(after).values() if t.requires_grad)
    loss_a.backward(); loss_b.backward()
    grads = 0; grad_error = 0.
    for key, parameter in old.named_parameters():
        other = dict(new.named_parameters())[key]
        assert (parameter.grad is None) == (other.grad is None), key
        if parameter.grad is not None:
            diff, _ = compare(parameter.grad, other.grad)
            grad_error = max(grad_error, diff); grads += 1
    optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4)
    optimizer.step()
    new.eval()
    artifact = save_model(new, output/name, kwargs=kwargs,
                          training={"method":"AdamW", "steps":1, "dataset":"synthetic migration verification"})
    loaded = load_tensor_artifact(artifact.path, trusted=True, warn_untrained=False)
    with torch.no_grad():
        reload_error, _ = compare(call(new), call(loaded.module))
    return {"name":name,"mapped_tensors":len(source),"output_tensors":count,"eval_max_abs":error,
            "train_max_abs":train_error,"gradient_tensors":grads,"gradient_max_abs":grad_error,
            "reloaded_max_abs":reload_error,"optimizer_steps":1,
            "mapping":{key:key for key in source},"weights_sha256":artifact.manifest['weights']['sha256']}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT/'weights/migration-validation')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(91)
    baseline=json.loads((ROOT/'lightnest/architecture/wgt/baseline.json').read_text())
    scratch=Path('/tmp/lightnest'); scratch.mkdir(exist_ok=True)
    reports=[]
    with tempfile.TemporaryDirectory(dir=scratch,prefix='wgt-reference-') as tmp:
        package=Path(tmp)/'_lightnest_baseline'; architecture=package/'architecture'; architecture.mkdir(parents=True)
        (package/'__init__.py').write_text(''); (architecture/'__init__.py').write_text('')
        for path in (ROOT/'lightnest/architecture').glob('*.py'):
            if path.name=='weightlang_io.py': continue
            relative=str(path.relative_to(ROOT))
            source=subprocess.check_output(['git','show',f"{baseline['commit']}:{relative}"],cwd=ROOT,text=True)
            import hashlib
            if relative in baseline['sha256']:
                assert hashlib.sha256(source.encode()).hexdigest()==baseline['sha256'][relative]
            (architecture/path.name).write_text(source.replace('lightnest.architecture','_lightnest_baseline.architecture'))
        sys.path.insert(0,tmp)
        def pair(stem,name,kwargs):
            a=getattr(importlib.import_module('_lightnest_baseline.architecture.'+stem),name)(**kwargs)
            b=getattr(importlib.import_module('lightnest.architecture.'+stem),name)(**kwargs)
            return a,b
        def case(label,stem,name,kwargs,call):
            a,b=pair(stem,name,kwargs); reports.append(check_case(label,a,b,call,kwargs,args.output))
        from lightnest.architecture.generate import condition_tensor
        previous=torch.tensor([[256,1,2,3],[256,7,8,9]])
        condition=condition_tensor([{'exit':7},{'stdout':'0102030405060708'}])
        for depth in (0,2,4):
            case(f'generator-depth-{depth}','generate','ELFGenerator',{'width':16,'capacity':32},
                 lambda m,d=depth:m(previous,condition,red_depth=d))
        def stream(m):
            first=m(previous[:,:2],condition,red_depth=2)
            return m(previous[:,2:],condition,red_depth=2,hidden=first['hidden'],position_offset=2)
        case('generator-stream','generate','ELFGenerator',{'width':16,'capacity':32},stream)
        for mode in ('dynamic','fixed','dense'):
            case('hnet-'+mode,'hnet','HNetELFGenerator',{'width':16,'capacity':32,'chunk_mode':mode},stream)
        from lightnest.architecture.model import RGBConfig
        config=RGBConfig(width=16,chunk_size=8,heads=2,landmarks=2,detail_k=2,red_depths=(2,4),note_slots=2)
        for mode in ('routed','dense','compressed'):
            case('rgb-'+mode,'model','RGBPolicy',{'config':config},lambda m,mode=mode:m(b'abcabcabxyz','edit',depth=4,mode=mode))
        from lightnest.architecture.any2any import ArtifactRequest
        requests=[ArtifactRequest('text','understand'),ArtifactRequest('raw','edit','x')]
        def any_forward(m):
            encoded,valid,kinds=m.encode_batch([b'abcdef',b'xy'],requests)
            encoded,notes=m.contextualize(encoded)
            abstract,guidance=m.think(encoded,depth=4)
            return {'abstract':abstract,'heads':m.heads(encoded,guidance),'notes':notes,'valid':valid}
        for encoding in ('mean','gru'):
            case('any2any-'+encoding,'any2any','ArtifactCore',{'width':32,'heads':4,'sequence_limit':256,'note_slots':2,'request_encoding':encoding},any_forward)
        for size in (4,8,16):
            case('semantic-chunk-'+str(size),'any2any','ArtifactCore',{'width':32,'heads':4,'sequence_limit':256,'note_slots':2,'semantic_chunk_size':size},lambda m:m.semantic_embedding([b'abcdef',b'xy']))
        from lightnest.architecture.byte_tokenizer import ByteBPE
        tokenizer=ByteBPE.train([b'return value + 1\n'*8, b'return smaller value\n'*8],vocab_size=272)
        def language(m):
            memory=m.semantic_tokens([b'return value + 1',b'value'])
            previous=torch.tensor([[tokenizer.bos,1],[tokenizer.bos,2]])
            result,hidden=m.language_forward(memory,previous)
            next_result,next_hidden=m.language_forward(memory,previous[:,:1],hidden)
            return {'memory':memory,'result':result,'hidden':hidden,'next':next_result,'next_hidden':next_hidden}
        case('semantic-bpe-language','any2any','ArtifactCore',{'width':32,'heads':4,'sequence_limit':256,'note_slots':2,'semantic_bpe':tokenizer.serialized},language)
        case('x86-world','x86_world','X86World',{'condition_width':4,'candidates':1,'image_capacity':64,'memory_size':256,'output_size':8},
             lambda m:m(torch.ones(1,4),torch.zeros(1,8,dtype=torch.long),torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),steps=2,rounds=2))
        for name in ('StructureNet','TokenStructureNet','ASTStructureNet'):
            case(name,'structure',name,{'width':16},lambda m:m(b'def hello(x): return x+1\n'))
        case('openat','linux_environment','LearnedOpenat',{'path_size':8,'width':8},lambda m:m(torch.tensor([[47,97,0,3,0,0,0,0]]),torch.tensor([2])))
    report={'baseline_commit':baseline['commit'],'device':'cpu','dtype':'float32','atol':1e-6,'rtol':1e-5,'cases':reports}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(reports),'max_abs':max(r['eval_max_abs'] for r in reports),'report':str(args.output/'report.json')}))


if __name__=='__main__': main()
