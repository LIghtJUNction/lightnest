"""Frozen comparisons for evidence-backed partial repair, no neural training.

Use an external directory built by compiled_v3.run.build. Do not use held-out
scores to pick a checkpoint, seed or mutation. Timing excludes compilation and
final auditing and includes search preparation, witnesses, failed trials and
fallback. Each request begins from the same unmodified machine-code input.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
from experiments.compiled_v3.run import load_suite, dump
from experiments.compiled_v3.search import Request, repair
from experiments.compiled_v3.vm import NativeVM, pack, python_run
from .repair import repair_bridge

MODES = ('single', 'random_pair', 'random_memory', 'bridge_only', 'bridge')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): sha(p)
        for folder in ('compiled_v3', 'repair_memory_v1')
        for p in sorted((root/'experiments'/folder).iterdir())
        if p.suffix in ('.py', '.c')}


def summarize(rows):
    result = defaultdict(Counter)
    for r in rows:
        for group in ('all', r['kind']):
            result[r['mode']+'/'+group].update(
                tasks=1, broken=int(r['broken']), repaired=int(r['broken'] and r['passed']),
                passed=int(r['passed']), false_accepted=int(r['status']=='candidate' and not r['passed']),
                clean_changed=int(r['kind']=='clean' and bool(r['patches'])),
                tried=r['tried'], vm_calls=r['vm_calls'], vm_steps=r['vm_steps'],
                search_seconds=r['seconds'])
    return dict(result)


def evaluate(root: Path, protocol_path: Path):
    protocol = json.loads(protocol_path.read_text())
    if source_hashes() != protocol['source_sha256']:
        raise ValueError('Source changed after protocol freeze')
    freeze = dict(utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(protocol_path), dataset_sha256=sha(root/'dataset.jsonl.gz'),
                  selected_mode=protocol['selected_mode'], before_first_evaluation=True)
    if (root/'memory-freeze.json').exists():
        raise FileExistsError('Do not overwrite an evaluation')
    dump(root/'memory-freeze.json', freeze)
    vm = NativeVM(root/'memory-runtime')
    records, witnesses, checks = [], {}, 0
    start = time.perf_counter()
    with gzip.open(root/'memory-predictions.jsonl.gz','xt') as output:
        for line in gzip.open(root/'dataset.jsonl.gz','rt'):
            build = json.loads(line)
            visible, expected = load_suite(build['generic'])
            hidden, h_expected = load_suite(build['hidden'])
            probes, p_expected = load_suite(build['original_probes'])
            audit_inputs, audit_expected = hidden+visible+probes, h_expected+expected+p_expected
            packed = pack(audit_inputs, audit_expected)
            score_cache = {}
            def score(code):
                if code not in score_cache:
                    score_cache[code] = vm.check(code, build['entry'], packed)
                return score_cache[code]
            for index, task in enumerate(build['tasks']):
                code = bytes.fromhex(task['code'])
                broken = score(code) >= 0
                request = Request(code, build['entry'], tuple(visible), tuple(expected))
                per_task = {}
                offset = index % len(MODES)
                order = MODES[offset:]+MODES[:offset]
                for mode in order:
                    tick = time.perf_counter()
                    decision = repair_bridge(request, vm, mode=mode,
                                             budget=protocol['budget'], width=protocol['width'])
                    seconds = time.perf_counter()-tick
                    failure = score(decision.code)
                    row = {'id':task['id'], 'family':build['family'], 'variant':build['variant'],
                        'kind':task['kind'], 'optimization':build['optimization'], 'mode':mode,
                        'broken':broken, 'passed':failure<0, 'audit_failure':failure,
                        'final_sha256':hashlib.sha256(decision.code).hexdigest(),
                        'final':decision.code.hex(), 'seconds':seconds,
                        **{k:v for k,v in asdict(decision).items() if k!='code'}}
                    records.append(row);per_task[mode]=row
                    output.write(json.dumps(row,separators=(',',':'))+'\n')
                    # One independent witness for EVERY distinct final image:
                    # its first failing case, or the first hidden case on success.
                    k = failure if failure >= 0 else 0
                    inp = audit_inputs[k]
                    key = row['final_sha256']+hashlib.sha256(bytes(inp)).hexdigest()
                    if key not in witnesses:
                        actual,_ = vm.run(decision.code, build['entry'], inp)
                        witnesses[key] = dict(id=task['id'],mode=mode,entry=build['entry'],
                            code=decision.code.hex(),input=inp.record(),actual=actual.record(),
                            expected=audit_expected[k].record())
                original = repair(request, vm, guided=True, budget=protocol['budget'])
                single = per_task['single']
                assert (original.code.hex(),original.status,original.tried) == (single['final'],single['status'],single['tried'])
                for mode in MODES:
                    r=per_task[mode]
                    assert r['tried']<=protocol['budget'] and len(r['patches'])<=2
                    if single['status'] in ('candidate','unchanged'):
                        assert (r['final'],r['status'],r['tried'])==(single['final'],single['status'],single['tried'])
                checks += 1
            output.flush()
            print('EVAL',build['build'],checks,'tasks',flush=True)
    dump(root/'memory-summary.json', dict(stats=summarize(records), tasks=checks,
        decisions=len(records), baseline_preservation_checks=checks,
        elapsed_including_audit_seconds=time.perf_counter()-start,
        protocol_sha256=sha(protocol_path), predictions_sha256=sha(root/'memory-predictions.jsonl.gz')))
    with gzip.open(root/'memory-witnesses.jsonl.gz','xt') as f:
        for witness in witnesses.values():
            f.write(json.dumps(witness,separators=(',',':'))+'\n')
    print('SUMMARY',json.dumps(summarize(records)),flush=True)


def audit(root: Path):
    from experiments.compiled_v3.run import decode_input, decode_output
    rows = [json.loads(s) for s in gzip.open(root/'memory-predictions.jsonl.gz','rt')]
    summary = json.loads((root/'memory-summary.json').read_text())
    assert summarize(rows)==summary['stats']
    checked,steps = 0,0
    for line in gzip.open(root/'memory-witnesses.jsonl.gz','rt'):
        w=json.loads(line)
        actual=python_run(bytes.fromhex(w['code']),w['entry'],decode_input(w['input']))
        expected=decode_output(w['actual'])
        assert actual.observable()==expected.observable() and actual.steps==expected.steps,(w['id'],w['mode'])
        checked+=1;steps+=actual.steps
    dump(root/'memory-audit.json',dict(passed=True,decisions_recounted=len(rows),
        independent_python_executions=checked,independent_python_steps=steps,
        scope='Distinct output witnesses, not every hidden input; separate decoder in same session.'))
    print('AUDIT',checked,steps,flush=True)


def timing(root: Path, protocol_path: Path):
    protocol=json.loads(protocol_path.read_text())
    if source_hashes()!=protocol['source_sha256']:
        raise ValueError('Source changed before timing')
    requests=[]
    for line in gzip.open(root/'dataset.jsonl.gz','rt'):
        b=json.loads(line);xs,ys=load_suite(b['generic'])
        for task in b['tasks']:
            requests.append((task['id'],Request(bytes.fromhex(task['code']),b['entry'],tuple(xs),tuple(ys))))
    reference={(r['id'],r['mode']):(r['final_sha256'],r['status'],r['tried'])
        for r in map(json.loads,gzip.open(root/'memory-predictions.jsonl.gz','rt'))}
    vm=NativeVM(root/'timing-runtime');raw=[]
    for repeat in range(5):
        order=('random_pair','bridge') if repeat%2==0 else ('bridge','random_pair')
        for mode in order:
            tick=time.perf_counter();cpu=time.process_time()
            for task_id,req in requests:
                d=repair_bridge(req,vm,mode=mode,budget=protocol['budget'],width=protocol['width'])
                assert (hashlib.sha256(d.code).hexdigest(),d.status,d.tried)==reference[task_id,mode]
            r=dict(repeat=repeat,mode=mode,wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu)
            raw.append(r);print('TIMING',r,flush=True)
    med={m:statistics.median(r['wall_seconds'] for r in raw if r['mode']==m) for m in ('random_pair','bridge')}
    dump(root/'memory-timing.json',dict(raw=raw,median_seconds=med,speedup=med['random_pair']/med['bridge'],
        requests_per_repeat=len(requests),repeated_decisions=len(requests)*10,
        excludes=['data loading','compilation','independent final audit'],
        includes=['candidate generation','sorting','witness traces','failed candidates','fallback','signature check'],
        identical_outputs_between_methods=False,notes='Different search outcomes; not lossless neural inference speed.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('evaluate','audit','timing'))
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--protocol',type=Path)
    a=parser.parse_args()
    if a.phase=='audit':
        audit(a.root)
    else:
        if a.protocol is None:parser.error('--protocol required')
        (evaluate if a.phase=='evaluate' else timing)(a.root,a.protocol)

if __name__=='__main__':
    main()
