"""Build, freeze, compare, and independently audit compiler-level repair tests.

Run development first, then freeze the selected policy before evaluation.
All generated binaries, observations and per-task logs belong outside Git.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from collections import Counter, defaultdict
from .corpus import FAMILIES, source
from .inputs import random_inputs, probe_inputs, hidden_inputs, identity
from .search import POLICIES, Request, repair, edits, patched
from .vm import NativeVM, Input, Output, compile_kernel, reference, pack, python_run, run_command


def sha(data: bytes) -> str:return hashlib.sha256(data).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(path: Path, obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def encode_output(o):return o.record()
def decode_output(o):return Output(o['value'],(ct.c_uint32*32)(*o['a']),(ct.c_uint32*32)(*o['out']),o['status'],o['steps'])
def decode_input(i):return Input.make(i['a'],i['n'],i['key'],i['alias'])
def source_hashes():return {p.name:sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py','.c')}


def make_mutants(code: bytes, seed: int, single: int, double: int):
    rng=random.Random(seed);options=list(edits(code));rng.shuffle(options)
    chosen=[];seen={sha(code)}
    for pos,w in options:
        candidate=patched(code,pos,w);h=sha(candidate)
        if h not in seen:
            chosen.append({'kind':'single','code':candidate.hex(),'faults':[[pos,w]]});seen.add(h)
        if len(chosen)==single:break
    if len(chosen)!=single:raise ValueError('Not enough unique single faults')
    attempts=0
    while len(chosen)<single+double:
        attempts+=1
        if attempts>10000:raise ValueError('Not enough unique double faults')
        first,second=rng.sample(options,2)
        if first[0]==second[0]:continue
        candidate=patched(patched(code,*first),*second);h=sha(candidate)
        if h in seen:continue
        chosen.append({'kind':'double','code':candidate.hex(),'faults':[list(first),list(second)]});seen.add(h)
    return [{'kind':'clean','code':code.hex(),'faults':[]}]+chosen


def build(root: Path, phase: str, variants: list[int], single=12, double=4):
    root.mkdir(parents=True,exist_ok=False);vm=NativeVM(root/'runtime');start=time.perf_counter()
    manifest=[];audit=[];reference_checks=0;unique_images=set();unique_tasks=set();lengths=[];max_steps=0
    dataset_path=root/'dataset.jsonl.gz'
    with gzip.open(dataset_path,'wt') as stream:
        for family in FAMILIES:
            for variant in variants:
                for opt in ('O0','O2','Os'):
                    name=f'{family}-p{variant}-{opt}';seed=int(sha(name.encode())[:16],16)
                    text=source(family,variant);code,entry,native=compile_kernel(text,root/'build'/name,opt)
                    unique_images.add(sha(code));lengths.append(len(code))
                    mutants=make_mutants(code,seed^0x55f2a197,single,double)
                    generic=random_inputs(seed^0x139faf26,64);original_probes=probe_inputs(code,seed^0x139faf26)
                    task_probes=[probe_inputs(bytes.fromhex(m['code']),seed^0x139faf26) for m in mutants]
                    forbidden={identity(i) for i in generic+original_probes}
                    forbidden.update(identity(i) for suite in task_probes for i in suite)
                    hidden=hidden_inputs(code,seed^0x706df741,forbidden,256)
                    assert not ({identity(i) for i in hidden}&forbidden)
                    # All reference outputs are checked against a second compiler,
                    # different ISA, and independent execution, before any repairs.
                    cache={}
                    for suite in [generic,original_probes,hidden]+task_probes:
                        fresh=[i for i in suite if identity(i) not in cache]
                        expected=reference(native,fresh);actual=vm.batch(code,entry,fresh)
                        for inp,a,b in zip(fresh,actual,expected,strict=True):
                            if a.observable()!=b.observable():raise AssertionError(('reference mismatch',name,inp.record(),a.record(),b.record()))
                            cache[identity(inp)]=b;reference_checks+=1;max_steps=max(max_steps,int(a.steps))
                    for inp in hidden[:8]:
                        a,_=vm.run(code,entry,inp);b=python_run(code,entry,inp)
                        if a.observable()!=b.observable() or a.steps!=b.steps:raise AssertionError(('Python reference mismatch',name))
                        audit.append({'program':name,'input':identity(inp),'steps':int(a.steps)})
                    def suite_data(suite):
                        return {'inputs':[i.record() for i in suite], 'outputs':[encode_output(cache[identity(i)]) for i in suite]}
                    tasks=[]
                    for j,(mutant,probes) in enumerate(zip(mutants,task_probes,strict=True)):
                        mid=f'{name}-{j:02d}';unique_tasks.add(sha(code+bytes.fromhex(mutant['code'])))
                        tasks.append(dict(mutant,id=mid,probes=suite_data(probes)))
                    record={'build':name,'family':family,'variant':variant,'optimization':opt,
                        'source_sha256':sha(text.encode()),'reference_sha256':sha(code),'reference':code.hex(),
                        'entry':entry,'generic':suite_data(generic),'original_probes':suite_data(original_probes),
                        'hidden':suite_data(hidden),'tasks':tasks}
                    line=json.dumps(record,separators=(',',':'),allow_nan=False)
                    stream.write(line+'\n');stream.flush()
                    manifest.append({'build':name,'source_sha256':sha(text.encode()),'reference_sha256':sha(code),
                        'dataset_row_sha256':sha(line.encode()),'tasks':len(tasks),'bytes':len(code)})
                    print('BUILD',name,len(code),'tasks',len(tasks),flush=True)
    info={'phase':phase,'created_utc':now(),'variants':variants,'single':single,'double':double,
        'builds':len(manifest),'tasks':sum(m['tasks'] for m in manifest),'unique_reference_images':len(unique_images),
        'unique_reference_mutant_pairs':len(unique_tasks),'binary_bytes_min':min(lengths),'binary_bytes_max':max(lengths),
        'reference_native_crosschecks':reference_checks,'reference_python_crosschecks':len(audit),
        'max_reference_vm_steps':max_steps,'dataset_sha256':sha(dataset_path.read_bytes()),
        'source_sha256':source_hashes(),'elapsed_seconds':time.perf_counter()-start,
        'python':platform.python_version(),'compiler':run_command(['clang','--version']).splitlines()[0],
        'native_compiler':run_command(['gcc','--version']).splitlines()[0],
        'manifest':manifest,'reference_audit':audit}
    dump(root/'construction.json',info);print('CONSTRUCTION',json.dumps({k:v for k,v in info.items() if k not in ('manifest','reference_audit','source_sha256')}),flush=True)


def load_suite(suite):
    return [decode_input(x) for x in suite['inputs']],[decode_output(x) for x in suite['outputs']]


def summarize(records):
    stats={}
    for policy in POLICIES:
        rows=[r for r in records if r['policy']==policy]
        groups={}
        for kind in ('all','clean','single','double'):
            selected=[r for r in rows if kind=='all' or r['kind']==kind]
            groups[kind]={'tasks':len(selected),'broken':sum(r['broken'] for r in selected),
                'repaired':sum(r['broken'] and r['final_pass'] for r in selected),
                'final_pass':sum(r['final_pass'] for r in selected),
                'edits':sum(r['changed'] for r in selected),
                'incorrect_accepted_edits':sum(r['changed'] and not r['final_pass'] for r in selected),
                'clean_program_edits':sum(r['kind']=='clean' and r['changed'] for r in selected),
                'missed_faults_returned_unchanged':sum(r['broken'] and r['status']=='unchanged' for r in selected),
                'abstentions':sum(r['status'] in ('abstain','budget') for r in selected),
                'exact_reference_restores':sum(r['broken'] and r['exact_reference'] for r in selected),
                'candidate_trials':sum(r['tried'] for r in selected),
                'search_vm_calls':sum(r['vm_calls'] for r in selected),
                'search_vm_steps':sum(r['vm_steps'] for r in selected),
                'search_seconds':sum(r['seconds'] for r in selected)}
        stats[policy]=groups
    return stats


def evaluate(root: Path, selection: Path | None):
    construction=json.loads((root/'construction.json').read_text())
    if sha((root/'dataset.jsonl.gz').read_bytes())!=construction['dataset_sha256']:raise ValueError('Dataset changed')
    if source_hashes()!=construction['source_sha256']:raise ValueError('Source changed after construction')
    if construction['phase']=='evaluation' and selection is None:raise ValueError('Evaluation requires a frozen development selection')
    selected=json.loads(selection.read_text()) if selection else None
    if selected and selected['source_sha256']!=source_hashes():raise ValueError('Selection source mismatch')
    freeze={'frozen_utc':now(),'source_sha256':source_hashes(),'dataset_sha256':construction['dataset_sha256'],
        'selection_sha256':sha(selection.read_bytes()) if selection else None,'selected_policy':selected['selected_policy'] if selected else None}
    dump(root/'evaluation-freeze.json',freeze)
    vm=NativeVM(root/'evaluation-runtime');records=[];audits={};start=time.perf_counter();cpu_start=time.process_time()
    with gzip.open(root/'predictions.jsonl.gz','wt') as writer,gzip.open(root/'dataset.jsonl.gz','rt') as source_stream:
        for line in source_stream:
            build=json.loads(line);original=bytes.fromhex(build['reference']);entry=build['entry']
            generic,gold_generic=load_suite(build['generic']);orig_probe,gold_orig_probe=load_suite(build['original_probes'])
            hidden,gold_hidden=load_suite(build['hidden'])
            audit_inputs=hidden+generic+orig_probe;gold=gold_hidden+gold_generic+gold_orig_probe;packed_audit=pack(audit_inputs,gold)
            score_cache={}
            def score(code):
                h=sha(code)
                if h not in score_cache:score_cache[h]=vm.check(code,entry,packed_audit)
                return score_cache[h]
            for task_index,task in enumerate(build['tasks']):
                code=bytes.fromhex(task['code']);failure=score(code);broken=failure>=0
                probes,gold_probes=load_suite(task['probes']);seen_decisions={}
                # Rotated deterministic order reduces a fixed warm-cache advantage.
                order=list(POLICIES);rotation=task_index%len(order);order=order[rotation:]+order[:rotation]
                for policy in order:
                    cases,expected=(probes,gold_probes) if policy=='trace_probe64' else (generic,gold_generic)
                    n=4 if policy.endswith('4') and not policy.endswith('64') else 64
                    request=Request(code,entry,tuple(cases[:n]),tuple(expected[:n]))
                    t=time.perf_counter();decision=repair(request,vm,guided=policy.startswith('trace'))
                    seconds=time.perf_counter()-t;final_failure=score(decision.code)
                    record={'id':task['id'],'family':build['family'],'build':build['build'],'variant':build['variant'],
                        'optimization':build['optimization'],'kind':task['kind'],'policy':policy,'broken':broken,
                        'input_failure':failure,'changed':decision.patch is not None,'status':decision.status,
                        'patch':decision.patch,'final_pass':final_failure<0,'final_failure':final_failure,
                        'final_sha256':sha(decision.code),'exact_reference':decision.code==original,
                        'tried':decision.tried,'eligible':decision.eligible,'vm_calls':decision.vm_calls,
                        'vm_steps':decision.vm_steps,'seconds':seconds,'assurance':decision.assurance}
                    records.append(record);writer.write(json.dumps(record,separators=(',',':'))+'\n')
                    seen_decisions[policy]=decision
                    for check_index in set((0,final_failure if final_failure>=0 else 255)):
                        inp=audit_inputs[check_index];key=sha(decision.code)+identity(inp)
                        if key not in audits:
                            out,_=vm.run(decision.code,entry,inp)
                            audits[key]={'code':decision.code.hex(),'entry':entry,'input':inp.record(),'expected_vm':out.record(),
                                'first_task':task['id'],'policy':policy,'test_index':check_index}
                for a,b in (('all4','trace4'),('all64','trace64')):
                    if seen_decisions[a].status!='budget' and seen_decisions[b].status!='budget':
                        if seen_decisions[a].code!=seen_decisions[b].code:raise AssertionError(('Trace pruning changed selected patch',task['id'],a,b))
            writer.flush();print('EVALUATED',build['build'],len(records),'policy-decisions',flush=True)
    with gzip.open(root/'audit-cases.jsonl.gz','wt') as f:
        for case in audits.values():f.write(json.dumps(case,separators=(',',':'))+'\n')
    stats=summarize(records)
    result={'completed_utc':now(),'phase':construction['phase'],'selected_policy_before_evaluation':freeze['selected_policy'],
        'tasks':construction['tasks'],'policy_decisions':len(records),'stats':stats,
        'source_sha256':source_hashes(),'dataset_sha256':construction['dataset_sha256'],
        'prediction_sha256':sha((root/'predictions.jsonl.gz').read_bytes()),'unique_vm_audit_cases':len(audits),
        'native_vm_calls_total':int(vm.work[0]),'native_vm_steps_total':int(vm.work[1]),
        'elapsed_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu_start}
    dump(root/'summary.json',result)
    if construction['phase']=='development':
        viable=[p for p in POLICIES if stats[p]['all']['repaired']>0]
        best=min(viable,key=lambda p:(stats[p]['all']['incorrect_accepted_edits'], -stats[p]['all']['repaired'],stats[p]['all']['search_vm_steps']))
        dump(root/'selection.json',{'fixed_utc':now(),'selected_policy':best,'source_sha256':source_hashes(),
            'development_dataset_sha256':construction['dataset_sha256'],'ranking':'fewest incorrect accepted edits; most repairs; fewest VM instructions',
            'evaluation_model_queries':0})
    print('SUMMARY',json.dumps(result),flush=True)


def audit(root: Path):
    """Recompute every reported counter; cross-execute distinct output witnesses."""
    records=[]
    with gzip.open(root/'predictions.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
    summary=json.loads((root/'summary.json').read_text())
    if summarize(records)!=summary['stats']:raise AssertionError('Summary cannot be reconstructed')
    vm=NativeVM(root/'audit-runtime');count=0;steps=0;statuses=Counter();start=time.perf_counter()
    with gzip.open(root/'audit-cases.jsonl.gz','rt') as f:
        for line in f:
            case=json.loads(line);inp=decode_input(case['input']);code=bytes.fromhex(case['code'])
            expected=decode_output(case['expected_vm']);actual=python_run(code,case['entry'],inp)
            if actual.observable()!=expected.observable() or actual.steps!=expected.steps:
                raise AssertionError(('Independent VM mismatch',case['first_task'],case['policy'],actual.record(),expected.record()))
            count+=1;steps+=actual.steps;statuses[str(actual.status)]+=1
            if count%1000==0:print('AUDIT',count,flush=True)
    result={'completed_utc':now(),'summary_counters_match':True,'independent_python_executions':count,
        'independent_python_steps':steps,'status_counts':dict(statuses),'elapsed_seconds':time.perf_counter()-start,
        'all_formal_decisions_represented':len(records),'independence_limit':'Separate decoder implementation in same session, not third-party certification'}
    dump(root/'audit.json',result);print('AUDIT_RESULT',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','evaluate','audit'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['development','evaluation'])
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    if args.action=='build':
        if args.phase is None:parser.error('build requires --phase')
        build(args.root,args.phase,[0] if args.phase=='development' else [2,3,4,5],
              single=4 if args.phase=='development' else 12,double=2 if args.phase=='development' else 4)
    elif args.action=='evaluate':evaluate(args.root,args.selection)
    else:audit(args.root)

if __name__=='__main__':main()
