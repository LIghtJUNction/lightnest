"""Replay actions and recompute scores WITHOUT importing the study scorer.

The independent part is scoring/replay code. The second VM is inherited,
not a third independently developed execution engine or external audit.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import time
from . import BASE, ARTIFACT_ROOT
from .prepare import read, write, sha
from lightnest_data.vm import execute_b


def replay(ep, record):
    raw=bytearray.fromhex(ep['binary'])
    edits=0
    stopped=False
    for step,a in enumerate(record['actions']):
        assert a['step']==step
        assert not stopped
        if a['stop']:
            stopped=True
            continue
        edits+=1
        if step>=8:
            break
        p,v=a['position'],a['value']
        if not 0<=p<len(raw):
            break
        if 'before' in a:
            assert a['before']==raw[p]
        if raw[p]==v:
            break
        raw[p]=v
    assert bytes(raw).hex()==record['final']
    assert stopped==record['explicit_stop']
    false_edit=not ep['edits'] and edits>0
    assert bool(false_edit)==record['false_edit']
    values=[]
    for inp,expected in ep['cases']:
        try:
            actual=execute_b(bytes(raw),inp)
        except (ValueError,IndexError):
            actual=None
        values.append(actual==expected)
    strict=bool(bytes(raw).hex()==ep['target'] and stopped and all(values) and (ep['edits'] or edits==0))
    assert strict==record['strict']
    return strict,bool(ep['edits']),bool(false_edit),len(values)


def audit(root):
    start=time.process_time()
    pools={'development':{r['id']:r for r in read(BASE/'baseline_data/regression.jsonl')},
           'holdout':{r['id']:r for r in read(root/'holdout.jsonl')}}
    checked=vm_calls=0
    results={}
    for split,pool in pools.items():
        for path in sorted(root.glob(f'{split}-*.json')):
            obj=json.loads(path.read_text())
            summary=Counter()
            pairs=defaultdict(list)
            assert len(obj['records'])==len(pool)
            assert len({r['id'] for r in obj['records']})==len(pool)
            for record in obj['records']:
                ep=pool[record['id']]
                strict,need,false,calls=replay(ep,record)
                checked+=1
                vm_calls+=calls
                summary['total']+=1
                summary['strict_total']+=strict
                summary['strict_edits']+=strict and need
                summary['false_edits']+=false
                if need and ep.get('lane') is not None:
                    pairs[ep['pair_id']].append(strict)
            summary['pairs_both']=sum(len(p)==2 and all(p) for p in pairs.values())
            for key,value in summary.items():
                assert obj['summary']['all'][key]==value,(path,key,value,obj['summary']['all'][key])
            results[path.name]=dict(summary)
    sample_checks=[]
    for seed in (104729,130363):
        a=json.loads((root/f'train-{seed}-16.json').read_text())
        b=json.loads((root/f'train-{seed}-64.json').read_text())
        assert [s['sample_indices'] for s in a['steps']]==[s['sample_indices'] for s in b['steps']]
        sample_checks.append({'seed':seed,'equal_sampling_stream':True,'updates_each':len(a['steps'])})
    frozen=json.loads((root/'final-freeze.json').read_text())
    for name,digest in frozen['files'].items():
        assert sha(root/name)==digest
    source_root=Path(__file__).parents[2]
    for name,digest in frozen['source'].items():
        assert sha(source_root/name)==digest
    original=json.loads((ARTIFACT_ROOT/'FILES.sha256.json').read_text())
    for name,meta in original['files'].items():
        assert sha(ARTIFACT_ROOT/name)==meta['sha256'],name
    out={'scoring_records_replayed':checked,'second_vm_calls':vm_calls,'all_counts_match':True,
         'exactly_frozen_sources_and_heads':True,'original_manifest_files_unchanged':len(original['files']),
         'training_sampling_checks':sample_checks,'summaries':results,
         'cpu_seconds':time.process_time()-start,'external_audit':False}
    write(root/'audit.json',out)
    print('AUDIT',checked,'sequences;',vm_calls,'second-VM calls; all match',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
