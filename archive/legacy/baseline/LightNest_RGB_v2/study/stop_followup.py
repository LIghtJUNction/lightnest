"""Post-primary repair of the gate; new holdout, unchanged model weights.

Only g4 (the original RGB v0 depth) is taken forward, BOTH seeds. No best-seed
selection. Public regression now explicitly becomes calibration information.
The first-stage test NEVER selects the new threshold. Its inputs are used
only for a no-overlap check while constructing the new holdout.
"""
from __future__ import annotations
import argparse,json,hashlib,time
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
import torch
from rgb.runtime import common,configure_torch
from rgb.stop_gate import load_stop_only,StopOnlyPolicy
from lightnest_data.core import file_sha,neutral_fingerprint
from lightnest_data.generator import generate_pair,FAMILIES
from lightnest_data.vm import execute_a,execute_b
from .prepare import simple_pair,write_rows
from .cache import read_rows
from .evaluate import action,Live,rollout,summarize,compare
THRESHOLDS=[.005,.01,.025,.05,.1,.2]


def prepare(root):
    out=root/'followup';out.mkdir(exist_ok=False)
    known={neutral_fingerprint(bytes.fromhex(r[k])) for name in ['fit','router_fit','calibration','test','regression'] for r in read_rows(root/'data'/f'{name}.jsonl') for k in ('binary','target')}
    cases=read_rows(root/'data/calibration.jsonl')[0]['cases'];xs=[x for x,y in cases]
    rows=[];excluded=0
    for fi,fam in enumerate(('simple',)+FAMILIES+('branch_spill','loop_spill')):
        want=32 if fam=='simple' else 8;i=0;ok=0
        while ok<want:
            seed=1251019001+fi*100000+i;i+=1
            rr=simple_pair(seed,ok%2,'confirm') if fam=='simple' else generate_pair(fam,seed,ok%2,(3,8),'confirm')
            fps={neutral_fingerprint(bytes.fromhex(r[k])) for r in rr for k in ('binary','target')}
            if known&fps:excluded+=1;continue
            known|=fps
            for r in rr:
                r.pop('source',None);r['cases']=[]
                for x in xs:
                    y=execute_a(bytes.fromhex(r['target']),x);assert y==execute_b(bytes.fromhex(r['target']),x)
                    assert execute_a(bytes.fromhex(r['binary']),x)==execute_b(bytes.fromhex(r['binary']),x)
                    r['cases'].append([x,y])
            rows+=rr;ok+=1
    write_rows(out/'confirm.jsonl',rows)
    protocol={'phase':'post-primary STOP-only follow-up','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'arms':['g4-104729','g4-130363'],'reason':'original4-round RGB path; keep both seeds; primary stage preserved',
        'weights_unchanged':True,'thresholds':THRESHOLDS,'calibration_sources':['data/calibration.jsonl','data/regression.jsonl'],
        'forbidden_for_selection':'data/test.jsonl (consumed primary test); followup/confirm.jsonl',
        'selection':'no per-case loss of red successes on BOTH public sets, no added false edit, >=16 blue episodes; minimize estimated rounds; tie lower threshold; else disable',
        'confirm_rows':len(rows),'confirm_groups':len(rows)//4,'excluded_collisions':excluded,
        'confirm_sha256':file_sha(out/'confirm.jsonl'),'model_queries_before_freeze':0,
        'method':'calibration can simulate truncations of known red paths; final and regression validate with real dispatch',
        'scope':'new seeds of existing templates; same-session empirical gate; no population risk certificate',
        'source_sha256':{'rgb/stop_gate.py':file_sha(root/'rgb/stop_gate.py'),'study/stop_followup.py':file_sha(Path(__file__))},
        'models':{tag:file_sha(root/'runs'/tag/'model.pt') for tag in ['g4-104729','g4-130363']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('FOLLOWUP_FROZEN',len(rows),flush=True)


def prefix_calibration(m,eps,red_records):
    """All possible STOP-only results are prefixes of the immutable red path."""
    by={e['id']:e for e in eps};unique={};paths={}
    for rec in red_records:
        e=by[rec['id']];code=bytearray.fromhex(e['binary']);path=[]
        for a in rec['actions']:
            key=(e['command'],bytes(code).hex());unique[key]={'command':key[0],'binary':key[1]};path.append((key,a,bytes(code).hex()))
            if a['stop'] or a['step']==8:break
            if not 0<=a['position']<len(code):break
            if a.get('before')==a['value']:break
            code[a['position']]=a['value']
        paths[e['id']]=path
    pred={};keys=list(unique)
    with torch.inference_mode():
        for start in range(0,len(keys),32):
            kk=keys[start:start+32];rr=[unique[k] for k in kk]
            _,_,ctx,h=m.encode_shared(*common.encode(rr,m.red.config));_,(sc,bi),risk=m.fast_readout(ctx,h)
            for j,key in enumerate(kk):pred[key]=(int(sc[j].argmax())==m.none_index,float(risk[j].sigmoid()))
    choices=[]
    for t in THRESHOLDS:
        lost=[];added=[];used=0;rounds=0
        for rec in red_records:
            e=by[rec['id']];strict=rec['strict'];nprop=0
            for key,a,current in paths[e['id']]:
                stop,risk=pred[key]
                if stop and risk<=t:
                    strict=(current==e['target'] and (bool(e['edits']) or nprop==0));used+=1;rounds+=m.config.blue_rounds;break
                rounds+=64
                if not a['stop']:nprop+=1
            if rec['strict'] and not strict:lost.append(e['id'])
            if not rec['strict'] and strict:added.append(e['id'])
        choices.append({'threshold':t,'lost':lost,'added':added,'blue_episodes':used,'round_examples':rounds})
    return choices,len(unique)


def calibrate(root,tag):
    configure_torch();m=load_stop_only(root/'runs'/tag/'model.pt');out=root/'followup'/tag;out.mkdir(exist_ok=False)
    decisions=[]
    for name,redfile in [('calibration',root/'calibration'/tag/'red_only.json'),('regression',root/'final'/tag/'regression-red_only.json')]:
        eps=read_rows(root/'data'/f'{name}.jsonl');red=json.loads(redfile.read_text())['records'];cur,n=prefix_calibration(m,eps,red)
        decisions.append({'dataset':name,'unique_prefix_states':n,'choices':cur})
    valid=[]
    for i,t in enumerate(THRESHOLDS):
        xs=[d['choices'][i] for d in decisions]
        if all(not x['lost'] for x in xs) and sum(x['blue_episodes'] for x in xs)>=16:
            valid.append((sum(x['round_examples'] for x in xs),t))
    threshold=min(valid)[1] if valid else .005
    selection={'enabled':bool(valid),'threshold':threshold,'stop_only':True,'model_sha256':file_sha(root/'runs'/tag/'model.pt'),
               'public_calibration':decisions,'runtime_source_sha256':file_sha(root/'rgb/stop_gate.py'),'deployment_approved':False}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print('STOP_CALIBRATED',tag,bool(valid),threshold,flush=True)


def confirm(root,tag):
    configure_torch();frozen=json.loads((root/'followup/final-freeze.json').read_text());out=root/'followup'/tag
    assert file_sha(out/'selection.json')==frozen['selections'][tag]
    assert file_sha(root/'followup/confirm.jsonl')==frozen['test_sha256']
    assert file_sha(root/'rgb/stop_gate.py')==frozen['runtime_sha256']
    m=load_stop_only(root/'runs'/tag/'model.pt');sel=json.loads((out/'selection.json').read_text());m.policy=StopOnlyPolicy(sel['enabled'],sel['threshold'])
    results={}
    for dataset,filename in [('confirm',root/'followup/confirm.jsonl'),('regression',root/'data/regression.jsonl')]:
        eps=read_rows(filename)
        for mode in ('red_only','auto'):
            cpu=time.process_time();rs=rollout(Live(m,mode),eps);results[mode]=rs
            assert all(a['stop'] for r in rs for a in r['actions'] if a['blue'])
            (out/f'{dataset}-{mode}.json').write_text(json.dumps({'summary':summarize(rs),'records':rs,'cpu_seconds':time.process_time()-cpu}))
            print('STOP_CONFIRM',tag,dataset,mode,summarize(rs)['all'],flush=True)
        cmp=compare(results['red_only'],results['auto']);(out/f'{dataset}-comparison.json').write_text(json.dumps(cmp,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','calibrate','confirm']);p.add_argument('--tag');p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
    if a.phase=='prepare':prepare(a.root.resolve())
    else:(calibrate if a.phase=='calibrate' else confirm)(a.root.resolve(),a.tag)
