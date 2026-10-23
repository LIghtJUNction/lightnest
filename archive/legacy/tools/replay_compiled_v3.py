"""Replay recorded decisions with the latest source after defensive hardening.

Usage: python tools/replay_compiled_v3.py /path/evaluation /path/external
Reads only complete recorded studies. Writes one new replay receipt beside each.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compiled_v3 import search,vm
from experiments.compiled_v3.run import decode_input,decode_output,dump,sha,now,source_hashes


def suite(data):return tuple(decode_input(x) for x in data['inputs']),tuple(decode_output(x) for x in data['outputs'])


def replay(root: Path):
    expected={}
    with gzip.open(root/'predictions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);expected[(row['id'],row['policy'])]=row
    machine=vm.NativeVM(root/'latest-replay-runtime');count=0;t=time.perf_counter();changed=[]
    with gzip.open(root/'dataset.jsonl.gz','rt') as f:
        for line in f:
            build=json.loads(line);entry=build['entry']
            for task in build['tasks']:
                code=bytes.fromhex(task['code'])
                policies=search.POLICIES if 'family' in build else ('trace64','trace_probe64','trace_random128','trace_mixed128')
                for policy in policies:
                    if policy=='trace_probe64':data=task['probes']
                    elif policy=='trace_random128':data=build['random128']
                    elif policy=='trace_mixed128':data=task['mixed']
                    else:data=build['generic']
                    cases,gold=suite(data)
                    if policy in ('all4','trace4'):cases,gold=cases[:4],gold[:4]
                    decision=search.repair(search.Request(code,entry,cases,gold),machine,guided=policy.startswith('trace'))
                    old=expected[(task['id'],policy)]
                    patch=list(decision.patch) if decision.patch is not None else None
                    if (sha(decision.code),decision.status,patch,decision.tried)!=(old['final_sha256'],old['status'],old['patch'],old['tried']):
                        changed.append([task['id'],policy])
                    count+=1
    receipt={'completed_utc':now(),'decisions_replayed':count,'changed_decisions':changed,
        'current_core_source_sha256':source_hashes(),'elapsed_seconds':time.perf_counter()-t,
        'native_vm_calls':int(machine.work[0]),'native_vm_steps':int(machine.work[1]),
        'reason':'Include instruction bytes read as data in the conservative trace mask; do not redefine past scores.'}
    dump(root/'latest-source-replay.json',receipt)
    if changed:raise AssertionError(f'{len(changed)} changed decisions: see receipt')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit(__doc__)
    for arg in sys.argv[1:]:replay(Path(arg))
