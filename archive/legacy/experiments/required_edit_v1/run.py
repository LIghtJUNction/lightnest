"""Reproduce a small loss-only experiment from local, versioned LightNest assets.

No remote calls, candidate host execution, architecture changes, or automatic
promotion. Run prepare, train, evaluate and audit in that order. Outputs must
be outside the source tree. The prior Hybrid workspace supplies code and data;
expanded checkpoints are research parents, not the accepted default C64.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time
import zipfile
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('required_loss', HERE / 'loss.py')
lossmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(lossmod)
VARIANTS = lossmod.VARIANTS
HEADS = ('byte_fusion', 'pointer', 'replacement', 'stop_fusion', 'stop')
SEEDS = (104729, 130363)


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def read(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        for row in rows: f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')


def setup(workspace):
    base = workspace / 'baseline/LightNest_RGB_v2'
    sys.path.insert(0, str(base))
    from rgb.runtime import common, configure_torch
    from rgb.scripted_core import LoopCore
    from study.evaluate import rollout, summarize, compare, action
    from study.prepare import simple_pair
    from lightnest_data.generator import FAMILIES, generate_pair
    from lightnest_data.core import neutral_fingerprint, validate_program
    from lightnest_data.vm import execute_a, execute_b
    configure_torch()
    return locals()


def source_paths(workspace):
    base = workspace / 'baseline/LightNest_RGB_v2'
    return [p for folder in ('runtime', 'rgb', 'study', 'lightnest_data')
            for p in sorted((base / folder).rglob('*.py'))]


def prepare(a, lib):
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    for name in ('data', 'parents', 'models', 'train', 'evaluation'): (out / name).mkdir()
    fit = read(a.workspace / 'data/fit.jsonl')
    assert len(fit) == 3584 and len({r['id'] for r in fit}) == len(fit)
    jsonl(out / 'data/train.jsonl', fit)
    jsonl(out / 'data/regression.jsonl', read(lib['base'] / 'baseline_data/regression.jsonl'))
    jsonl(out / 'data/public.jsonl', read(lib['base'] / 'baseline_data/calibration.jsonl'))
    known, exact = set(), set()
    # The full v2 index, not only the sampled fit subset, guards new test groups.
    with zipfile.ZipFile(a.dataset_archive) as z:
        names = [n for n in z.namelist() if '/index/' in n and n.endswith('.jsonl.gz')]
        if not names: raise ValueError('expanded dataset indexes missing')
        for n in names:
            with gzip.GzipFile(fileobj=z.open(n)) as f:
                for line in f:
                    r = json.loads(line); known.add(r['neutral_sha256'])
                    exact.update((r['binary_sha256'], r['target_sha256']))
    def remember(rows):
        for r in rows:
            if not {'binary', 'target'} <= r.keys(): continue
            for k in ('binary', 'target'):
                b = bytes.fromhex(r[k]); known.add(lib['neutral_fingerprint'](b))
                exact.add(hashlib.sha256(b).hexdigest())
    for p in a.workspace.rglob('*.jsonl'):
        if any(d in p.parts for d in ('data', 'baseline_data')): remember(read(p))
    with zipfile.ZipFile(a.rgb1_archive) as z:
        for n in z.namelist():
            if n.endswith('.jsonl') and ('/data/' in n or n.endswith('/followup/confirm.jsonl')):
                remember(json.loads(line) for line in z.read(n).splitlines())
    rng = random.Random(710202026)
    xs = [[0,0,0],[1,2,3],[-1,2,-3],[2,5,3],[2147483647,1,-1],[-2147483648,-1,1],
          [7,7,0],[-13,13,5]] + [[rng.randint(-100000,100000) for _ in range(3)] for _ in range(8)]
    test, rejected, calls = [], 0, 0
    fams = ('simple',) + lib['FAMILIES'] + ('branch_spill', 'loop_spill')
    for length_name, bounds, seed_base in [('normal',(3,8),720260000),('stress',(9,12),730260000)]:
        for fi, family in enumerate(fams):
            got = i = 0
            while got < 8:
                seed = seed_base + fi * 100000 + i; i += 1
                prefix = 'required-edit-v1/' + length_name
                rs = (lib['simple_pair'](seed,got%2,prefix) if family == 'simple' else
                      lib['generate_pair'](family,seed,got%2,bounds,prefix))
                fps = {lib['neutral_fingerprint'](bytes.fromhex(r[k])) for r in rs for k in ('binary','target')}
                ex = {hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in rs for k in ('binary','target')}
                if fps & known or ex & exact: rejected += 1; continue
                known |= fps; exact |= ex
                for r in rs:
                    r.pop('source',None);r['difficulty'] = length_name;r['cases'] = []
                    for key in ('binary','target'): lib['validate_program'](bytes.fromhex(r[key]))
                    for x in xs:
                        for key in ('binary','target'):
                            ya = lib['execute_a'](bytes.fromhex(r[key]),x)
                            yb = lib['execute_b'](bytes.fromhex(r[key]),x);calls += 2
                            assert ya == yb
                            if key == 'target': r['cases'].append([x,ya])
                test.extend(rs);got += 1
    jsonl(out / 'data/test.jsonl', test)
    parents = {}
    for seed in SEEDS:
        src = a.parents / f'expanded-{seed}.pt'
        d = torch.load(src, map_location='cpu', weights_only=True)
        if d['rounds'] != 64 or d['step'] != 2048: raise ValueError('wrong research parent')
        path = out / f'parents/{seed}.pt'
        torch.save({'config':d['config'],'state_dict':d['state_dict'],'seed':seed,'rounds':64},path)
        parents[str(seed)] = {'original_sha256':sha(src),'parent_sha256':sha(path),
                             'status':'unpromoted expanded research checkpoint'}
    source = {str(p.resolve()):sha(p) for p in [HERE/'loss.py',HERE/'run.py',*source_paths(a.workspace)]}
    protocol = {'id':'LightNest-required-edit-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
                'source_commit':'7a387672105ef68b3117371e19e56819fbe7d6db',
                'seeds':list(SEEDS),'variants':list(VARIANTS),'steps':512,'batch':32,'lr':.0001,
                'weight_decay':.01,'gradient_clip':2.,'rounds':64,'train_heads':list(HEADS),
                'sampling':'8 legacy +24 hard; each half EDIT/STOP; same exact stream within each parent',
                'primary_hypothesis':'coverage improves complete multi-edit repair; weakest is aggressive contrast',
                'selection':'all fixed endpoints reported; no checkpoint/seed/threshold selection',
                'promotion':'none automatically; require no lost successes or new false edits and a repair gain in both parents',
                'test_rows':len(test),'test_groups':len({r['pair_id'] for r in test}),
                'test_scope':'new seeds of existing templates; simple family is a control, not a longer stress family',
                'test_model_calls_at_freeze':0,'collision_rejections':rejected,'construction_vm_calls':calls,
                'data':{p.name:sha(p) for p in (out/'data').glob('*.jsonl')},'parents':parents,
                'source':source,'environment':{'python':sys.version,'torch':torch.__version__,
                  'cuda':torch.cuda.is_available(),'threads':torch.get_num_threads()},
                'prior_archives':{p.name:sha(p) for p in (a.dataset_archive,a.rgb1_archive)},
                'disclosure':'label-free live inference; cached frozen training core only; no model architecture changes'}
    write(out/'protocol.json',protocol)
    print('PREPARED',len(fit),'train,',len(test),'test,',calls,'VM calls',flush=True)


def collate(cache, rows, common):
    es = [cache[r['id']] for r in rows]; length = max(e['binary'].shape[0] for e in es)
    def pad(key,n):
        x = es[0][key]; target = x.new_zeros((len(es),n,*x.shape[1:]))
        for j,e in enumerate(es): target[j,:len(e[key])] = e[key]
        return target
    b = pad('binary',length)
    ctx = common.arch.ReadoutContext(b,pad('emb',length),pad('slots',length),
                                    torch.stack([e['command'] for e in es]),b[:,::4].ne(0))
    return pad('h',length//4),ctx


def cache_states(m, rows, lib):
    loop = torch.jit.script(lib['LoopCore'](m).eval()); cache = {}; start = time.process_time()
    with torch.no_grad():
        for i in range(0,len(rows),32):
            es = rows[i:i+32]
            inputs,rel,ctx = m.encode_inputs(*lib['common'].encode(es,m.config))
            h = loop(inputs,inputs,rel,ctx.valid,64)
            for j,r in enumerate(es):
                length = len(r['binary'])//2
                cache[r['id']] = {'binary':ctx.binary[j,:length].clone(),'emb':ctx.byte_embeddings[j,:length].clone(),
                  'slots':ctx.slots[j,:length].clone(),'command':ctx.command[j].clone(),'h':h[j,:length//4].clone()}
    return cache, time.process_time()-start


def check_freeze(a):
    protocol = json.loads((a.out/'protocol.json').read_text())
    for p,h in protocol['source'].items():
        if sha(p) != h: raise ValueError('source changed after freeze: '+p)
    for p,h in protocol['data'].items():
        if sha(a.out/'data'/p) != h: raise ValueError('data changed')
    return protocol


def train(a, lib):
    cfg = check_freeze(a); rows = read(a.out/'data/train.jsonl'); common = lib['common']
    pools = {}
    for r in rows: pools.setdefault((r['family'],bool(r['edits'])),[]).append(r)
    hard_families = sorted({r['family'] for r in rows} - {'legacy'})
    all_logs = []
    for seed in SEEDS:
        initial = common.load_model(a.out/f'parents/{seed}.pt').eval()
        for p in initial.parameters(): p.requires_grad_(False)
        cache,cache_cpu = cache_states(initial,rows,lib)
        print('CACHED',seed,len(cache),round(cache_cpu,3),flush=True)
        initial_state = {k:v.clone() for k,v in initial.state_dict().items()}
        for variant in VARIANTS:
            torch.manual_seed(seed); rng = random.Random(seed+7011)
            model = deepcopy(initial)
            for name in HEADS:
                for p in getattr(model,name).parameters(): p.requires_grad_(True)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            model.train();log=[];counts=Counter();start=time.process_time();wall=time.perf_counter()
            for step in range(cfg['steps']):
                es = []
                for j in range(32):
                    fam = 'legacy' if j < 8 else rng.choice(hard_families)
                    es.append(rng.choice(pools[(fam,j%2==0)]))
                counts.update(r['id'] for r in es)
                h,ctx = collate(cache,es,common);sc,bi = model.decode_state(h,ctx)
                optimizer.zero_grad(set_to_none=True)
                loss,pl,bl = lossmod.required_edit_loss(sc,bi,es,model.none_index,variant)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip'])
                if not bool(torch.isfinite(norm)): raise FloatingPointError('gradient')
                optimizer.step()
                log.append({'step':step+1,'ids':[r['id'] for r in es],'loss':float(loss.detach()),
                            'position_loss':float(pl.detach()),'byte_loss':float(bl.detach()),'gradient_norm':float(norm)})
            cpu = time.process_time()-start
            frozen_equal = all(torch.equal(v,model.state_dict()[k]) for k,v in initial_state.items()
                               if k.split('.')[0] not in HEADS)
            assert frozen_equal
            state = {'config':model.config.__dict__,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'sampler_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'step':cfg['steps'],
                     'variant':variant,'seed':seed,'rounds':64,'counts':dict(counts),'protocol_sha256':sha(a.out/'protocol.json')}
            path = a.out/f'models/{variant}-{seed}.pt'
            with path.open('xb') as f: torch.save(state,f)
            jsonl(a.out/f'train/{variant}-{seed}.jsonl',log)
            summary={'variant':variant,'seed':seed,'updates':len(log),'presentations':sum(counts.values()),
              'unique_records':len(counts),'cpu_seconds':cpu,'wall_seconds':time.perf_counter()-wall,
              'cache_preparation_cpu_seconds':cache_cpu,'frozen_core_equal':frozen_equal,
              'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in trainable),
              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'checkpoint_sha256':sha(path),
              'order_sha256':hashlib.sha256(json.dumps([r['ids'] for r in log],separators=(',',':')).encode()).hexdigest(),
              'first_loss':log[0]['loss'],'last_loss':log[-1]['loss']}
            all_logs.append(summary);print('TRAINED',json.dumps(summary),flush=True)
        del cache
    write(a.out/'training-summary.json',all_logs)
    for seed in SEEDS:
        streams=[r['order_sha256'] for r in all_logs if r['seed']==seed]
        assert len(set(streams))==1
    write(a.out/'final-freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
          'test_model_calls':0,'checkpoints':{str(p.relative_to(a.out)):sha(p)
                for folder in ('parents','models') for p in (a.out/folder).glob('*.pt')},
          'protocol_sha256':sha(a.out/'protocol.json')})


def evaluate(a, lib):
    cfg=check_freeze(a); frozen=json.loads((a.out/'final-freeze.json').read_text())
    for p,h in frozen['checkpoints'].items():
        if sha(a.out/p)!=h: raise ValueError('checkpoint changed')
    summaries=[]
    for seed in SEEDS:
        comparisons={}
        for variant in ('parent',)+VARIANTS:
            path=a.out/(f'parents/{seed}.pt' if variant=='parent' else f'models/{variant}-{seed}.pt')
            model=lib['common'].load_model(path).eval();loop=torch.jit.script(lib['LoopCore'](model).eval())
            @torch.inference_mode()
            def predict(rows):
                if any(set(r)!= {'command','binary'} for r in rows): raise ValueError('label in inference')
                inp,rel,ctx=model.encode_inputs(*lib['common'].encode(rows,model.config))
                h=loop(inp,inp,rel,ctx.valid,64);sc,bi=model.decode_state(h,ctx)
                return [lib['action'](sc[j],bi[j],model.none_index)|{'blue':False,'rounds':64} for j in range(len(rows))]
            for split in ('public','regression','test'):
                es=read(a.out/f'data/{split}.jsonl');start=time.process_time()
                records=lib['rollout'](predict,es,batch=16);metrics=lib['summarize'](records)
                if variant=='parent': comparisons[split]=records
                comparison=lib['compare'](comparisons[split],records)
                receipt={'seed':seed,'variant':variant,'split':split,'summary':metrics,'comparison':comparison,
                         'cpu_seconds_including_execution':time.process_time()-start}
                write(a.out/f'evaluation/{variant}-{seed}-{split}.json',receipt|{'records':records})
                summaries.append(receipt)
                print('EVAL',seed,variant,split,metrics['all'],'lost',len(comparison['lost_success_ids']),flush=True)
    write(a.out/'summary.json',summaries)
    write(a.out/'test-consumption.json',{'consumed':True,'new_test_rows':cfg['test_rows'],
                'only_frozen_endpoints':True,'no_threshold_or_seed_selection':True,'stable_default_replaced':False})


def audit(a, lib):
    check_freeze(a); totals=Counter()
    for p in sorted((a.out/'evaluation').glob('*.json')):
        d=json.loads(p.read_text());es={r['id']:r for r in read(a.out/f'data/{d["split"]}.jsonl')}
        ids=[r['id'] for r in d['records']]
        assert len(ids)==len(set(ids))==len(es) and set(ids)==set(es)
        ct=Counter()
        for r in d['records']:
            e=es[r['id']];b=bytearray.fromhex(e['binary']);stopped=False;proposed=0
            for i,act in enumerate(r['actions']):
                assert not stopped and act['step']==i and i<9
                if act['stop']:stopped=True;break
                proposed+=1
                if i==8 or not 0<=act['position']<len(b):break
                assert b[act['position']]==act['before']
                if b[act['position']]==act['value']:break
                b[act['position']]=act['value']
            assert b.hex()==r['final'] and stopped==r['explicit_stop']
            ok=True
            for x,y in e['cases']:
                try:actual=lib['execute_b'](bytes(b),x)
                except (ValueError,IndexError):actual=None
                if actual!=y:ok=False
                totals['second_vm_calls']+=1
            strict=bool(b.hex()==e['target'] and stopped and ok and (e['edits'] or proposed==0))
            false=not e['edits'] and proposed>0
            assert strict==r['strict'] and false==r['false_edit'] and ok==r['numeric']
            ct['strict_total']+=strict;ct['false_edits']+=false;ct['strict_edits']+=bool(e['edits']) and strict
            totals['sequences']+=1
        assert all(d['summary']['all'][k]==v for k,v in ct.items())
        totals['evaluation_files']+=1
    write(a.out/'audit.json',{'passed':True,**totals,'scope':'separate metric reconstruction in same session; supplied second interpreter, not external certification'})
    print('AUDITED',dict(totals),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prepare','train','evaluate','audit'))
    ap.add_argument('--workspace',type=Path,required=True,help='extracted, verified Hybrid v1 archive root')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--parents',type=Path)
    ap.add_argument('--dataset-archive',type=Path)
    ap.add_argument('--rgb1-archive',type=Path)
    a=ap.parse_args();a.workspace=a.workspace.resolve();a.out=a.out.resolve()
    if a.out==a.workspace or a.out.is_relative_to(HERE): raise ValueError('separate output directory required')
    lib=setup(a.workspace)
    if a.phase=='prepare' and any(v is None for v in (a.parents,a.dataset_archive,a.rgb1_archive)):
        ap.error('prepare requires parents and both source archives')
    globals()[a.phase](a,lib)

if __name__=='__main__': main()
