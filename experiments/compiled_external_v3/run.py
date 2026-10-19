"""Real-source follow-up: zlib adler32_z, equal-budget random vs mixed tests.

This is a separately registered follow-up, not a rewrite of compiled-v3 results.
The adapted checksum body is upstream code; injected faults are artificial.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
from experiments.compiled_v3 import inputs, search, vm
from experiments.compiled_v3.run import make_mutants, dump, sha, now, decode_input, decode_output

POLICIES=('trace64','trace_probe64','trace_random128','trace_mixed128')


def unique(cases, count):
    result=[];seen=set()
    for case in cases:
        key=inputs.identity(case)
        if key not in seen:seen.add(key);result.append(case)
        if len(result)==count:return result
    raise ValueError('Insufficient distinct specification examples')


def source_hashes():
    result={}
    for folder in (Path(__file__).parent,Path(search.__file__).parent):
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.py','.c','.md'):result[str(path.relative_to(folder.parent))]=sha(path.read_bytes())
    return result


def main(root: Path, protocol: Path):
    root.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu=time.process_time()
    freeze={'fixed_utc':now(),'source_sha256':source_hashes(),'protocol_sha256':sha(protocol.read_bytes()),
            'previous_selection_retained':'trace_probe64 is NOT retroactively replaced in the first study'}
    dump(root/'freeze.json',freeze)
    machine=vm.NativeVM(root/'runtime');records=[];audits={};builds=[];reference_count=0
    text=Path(__file__).with_name('adler32.c').read_text()
    with gzip.open(root/'dataset.jsonl.gz','wt') as datafile,gzip.open(root/'predictions.jsonl.gz','wt') as predictions:
        for opt in ('O0','O2','Os'):
            code,entry,native=vm.compile_kernel(text,root/'build'/opt,opt)
            seed=int(sha(('external-zlib-v1.3.1-'+opt).encode())[:16],16)
            generic=inputs.random_inputs(seed,64)
            random128=unique(generic+inputs.random_inputs(seed^0x306ea717,256),128)
            mutants=make_mutants(code,seed^0x60244015,128,64)
            suites=[];forbidden={inputs.identity(i) for i in random128}
            for mutant in mutants:
                raw=bytes.fromhex(mutant['code']);probes=inputs.probe_inputs(raw,seed,64)
                mixed=unique(generic+inputs.probe_inputs(raw,seed,192),128)
                suites.append((probes,mixed));forbidden.update(inputs.identity(i) for i in probes+mixed)
            hidden=inputs.hidden_inputs(code,seed^0x69142035,forbidden,512)
            canonical=unique(inputs.probe_inputs(code,seed,192),128)
            audit_inputs=hidden+random128+canonical;cache={};coverage=set()
            for suite in [audit_inputs]+[part for pair in suites for part in pair]:
                fresh=[i for i in suite if inputs.identity(i) not in cache]
                expected=vm.reference(native,fresh);actual=machine.batch(code,entry,fresh)
                for inp,a,b in zip(fresh,actual,expected,strict=True):
                    if a.observable()!=b.observable():raise AssertionError(('Compiler/reference mismatch',opt,inp.record()))
                    external=zlib.adler32(struct.pack('<32I',*inp.a)[:inp.n*4],inp.key)
                    if a.value!=external:raise AssertionError(('System zlib mismatch',opt))
                    cache[inputs.identity(inp)]=b;reference_count+=1
            for inp in hidden:
                _,seen=machine.run(code,entry,inp,trace=True);coverage.update(seen)
            gold=[cache[inputs.identity(i)] for i in audit_inputs];packed=vm.pack(audit_inputs,gold)
            score_cache={}
            def score(raw):
                h=sha(raw)
                if h not in score_cache:score_cache[h]=machine.check(raw,entry,packed)
                return score_cache[h]
            def encoded(suite):
                return {'inputs':[i.record() for i in suite],'outputs':[cache[inputs.identity(i)].record() for i in suite]}
            task_data=[]
            for index,(task,(probes,mixed)) in enumerate(zip(mutants,suites,strict=True)):
                raw=bytes.fromhex(task['code']);failure=score(raw);broken=failure>=0
                choices={'trace64':generic,'trace_probe64':probes,'trace_random128':random128,'trace_mixed128':mixed}
                task_id=f'zlib-{opt}-{index:03d}'
                order=list(POLICIES);rotation=index%4;order=order[rotation:]+order[:rotation]
                for policy in order:
                    cases=choices[policy];expected=[cache[inputs.identity(i)] for i in cases]
                    request=search.Request(raw,entry,tuple(cases),tuple(expected))
                    t=time.perf_counter();decision=search.repair(request,machine,guided=True)
                    seconds=time.perf_counter()-t;final_failure=score(decision.code)
                    row={'id':task_id,'optimization':opt,'kind':task['kind'],'policy':policy,'broken':broken,
                         'input_failure':failure,'changed':decision.patch is not None,'patch':decision.patch,
                         'status':decision.status,'final_pass':final_failure<0,'final_failure':final_failure,
                         'final_sha256':sha(decision.code),'exact_reference':decision.code==code,
                         'tried':decision.tried,'eligible':decision.eligible,'vm_calls':decision.vm_calls,
                         'vm_steps':decision.vm_steps,'seconds':seconds,'distinct_visible_tests':len(cases),
                         'assurance':decision.assurance}
                    records.append(row);predictions.write(json.dumps(row,separators=(',',':'))+'\n')
                    for check_index in set((0,final_failure if final_failure>=0 else 511)):
                        inp=audit_inputs[check_index];key=sha(decision.code)+inputs.identity(inp)
                        if key not in audits:
                            out,_=machine.run(decision.code,entry,inp)
                            audits[key]={'code':decision.code.hex(),'entry':entry,'input':inp.record(),'expected_vm':out.record(),
                                'first_task':task_id,'policy':policy,'test_index':check_index}
                task_data.append(dict(task,id=task_id,probes=encoded(probes),mixed=encoded(mixed)))
            build={'optimization':opt,'reference_sha256':sha(code),'bytes':len(code),'entry':entry,
                'observed_instruction_locations':len(coverage),'image_words':len(code)//4,'tasks':len(mutants)}
            builds.append(build)
            datafile.write(json.dumps(dict(build,reference=code.hex(),tasks=task_data,hidden=encoded(hidden),
                generic=encoded(generic),random128=encoded(random128),canonical=encoded(canonical)),separators=(',',':'))+'\n')
            datafile.flush();predictions.flush();print('EXTERNAL',opt,len(code),len(records),'decisions',flush=True)
    stats={}
    for policy in POLICIES:
        stats[policy]={}
        for kind in ('all','clean','single','double'):
            rows=[r for r in records if r['policy']==policy and (kind=='all' or r['kind']==kind)]
            stats[policy][kind]={'tasks':len(rows),'broken':sum(r['broken'] for r in rows),
                'repaired':sum(r['broken'] and r['final_pass'] for r in rows),
                'incorrect_accepted_edits':sum(r['changed'] and not r['final_pass'] for r in rows),
                'clean_program_edits':sum(r['changed'] and r['kind']=='clean' for r in rows),
                'missed_faults_returned_unchanged':sum(r['broken'] and r['status']=='unchanged' for r in rows),
                'abstentions':sum(r['status'] in ('abstain','budget') for r in rows),
                'edits':sum(r['changed'] for r in rows),'final_pass':sum(r['final_pass'] for r in rows),
                'search_vm_calls':sum(r['vm_calls'] for r in rows),'search_vm_steps':sum(r['vm_steps'] for r in rows),
                'candidate_trials':sum(r['tried'] for r in rows),'search_seconds':sum(r['seconds'] for r in rows)}
    checked=0;audit_steps=0
    with gzip.open(root/'audit-cases.jsonl.gz','wt') as f:
        for case in audits.values():
            inp=decode_input(case['input']);expected=decode_output(case['expected_vm'])
            other=vm.python_run(bytes.fromhex(case['code']),case['entry'],inp)
            if other.observable()!=expected.observable() or other.steps!=expected.steps:raise AssertionError(('Independent VM mismatch',case['first_task']))
            f.write(json.dumps(case,separators=(',',':'))+'\n');checked+=1;audit_steps+=other.steps
    if source_hashes()!=freeze['source_sha256']:raise AssertionError('Source changed during follow-up')
    result={'completed_utc':now(),'builds':builds,'stats':stats,'tasks':sum(b['tasks'] for b in builds),
        'policy_decisions':len(records),'trusted_reference_and_system_zlib_checks':reference_count,
        'independent_python_executions':checked,'independent_python_steps':audit_steps,
        'native_vm_calls_total':int(machine.work[0]),'native_vm_steps_total':int(machine.work[1]),
        'system_zlib_version':zlib.ZLIB_RUNTIME_VERSION,'elapsed_seconds':time.perf_counter()-started,
        'cpu_seconds':time.process_time()-cpu,'source_sha256':source_hashes(),
        'dataset_sha256':sha((root/'dataset.jsonl.gz').read_bytes()),
        'predictions_sha256':sha((root/'predictions.jsonl.gz').read_bytes())}
    dump(root/'summary.json',result);print('EXTERNAL_SUMMARY',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();main(a.root,a.protocol)
