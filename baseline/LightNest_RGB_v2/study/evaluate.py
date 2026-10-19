"""Full-sequence evaluation; prediction only sees request + current bytes.

Calibration may memoize exact frozen-input next actions. Final tests use real
subset dispatch with no proposal cache. No target, execution result, group ID,
or correct offset reaches either inference path. No native binary execution.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from rgb.network import load_rgb,RoutePolicy
from rgb.runtime import common,configure_torch
from lightnest_data.vm import execute_a,execute_b
from lightnest_data.core import file_sha
from .cache import read_rows


def action(sc,bi,none):
    p=int(sc.argmax());v=None if p==none else sum(int(bi[p,k]>=0)<<k for k in range(8))
    return {'position':p,'value':v,'confidence':float(sc.softmax(-1)[p]),'stop':p==none}

class Live:
    def __init__(self,m,mode):self.m=m;self.mode=mode
    @torch.inference_mode()
    def __call__(self,rows):
        o=self.m(*common.encode(rows,self.m.red.config),mode=self.mode)
        return [action(o['scores'][j],o['bits'][j],self.m.none_index)|
                {'blue':bool(o['used_blue'][j]),'risk':float(o['predicted_blue_error'][j]),
                 'rounds':self.m.config.blue_rounds if bool(o['used_blue'][j]) else self.m.config.red_rounds,
                 'fallback_nonfinite':bool(o['fast_nonfinite_fallback'][j])} for j in range(len(rows))]

class Memo:
    """Calibration ONLY, never used by latency benchmark or final neural runs."""
    def __init__(self,m):self.m=m;self.cache={};self.mode='red_only';self.threshold=0.;self.neural_rows=0
    @torch.inference_mode()
    def __call__(self,rows):
        missing=[];seen=set()
        for r in rows:
            key=(r['command'],r['binary'])
            if key not in self.cache and key not in seen:missing.append(r);seen.add(key)
        if missing:
            o=self.m.forward_train(*common.encode(missing,self.m.red.config));self.neural_rows+=len(missing)
            for j,r in enumerate(missing):
                self.cache[(r['command'],r['binary'])]={k:action(o[k][0][j],o[k][1][j],self.m.none_index) for k in ('blue','red')}|{'risk':float(o['risk_logits'][j].sigmoid())}
        outs=[]
        for r in rows:
            d=self.cache[(r['command'],r['binary'])]
            use=self.mode=='blue_only' or (self.mode=='auto' and d['risk']<=self.threshold)
            outs.append(d['blue' if use else 'red']|{'blue':use,'risk':d['risk'],'rounds':self.m.config.blue_rounds if use else 64,'fallback_nonfinite':False})
        return outs


def score_episode(e,st):
    code=st['code'];exact=code.hex()==e['target'];need=bool(e['edits']);proposed=sum(not a['stop'] for a in st['actions'])
    first=None;agreement=True;numeric=True
    for x,y in e['cases']:
        vals=[];errors=[]
        for fn in (execute_a,execute_b):
            try:vals.append(fn(code,x));errors.append(None)
            except (ValueError,IndexError) as err:vals.append(None);errors.append(str(err))
        if vals[0]!=vals[1]:agreement=False
        if vals[0]!=y or vals[1]!=y:
            numeric=False
            if first is None:first={'input':x,'expected':y,'actual':vals,'errors':errors}
    return dict(id=e['id'],pair_id=e['pair_id'],family=e['family'],lane=e.get('lane'),need_edit=need,
        strict=bool(exact and st['stop'] and numeric and (need or proposed==0)),exact=exact,numeric=numeric,
        false_edit=not need and proposed>0,proposed=proposed,explicit_stop=st['stop'],status=st['status'],
        vm_agreement=agreement,execution_calls=2*len(e['cases']),first_failure=first,final=code.hex(),
        initial_sha256=hashlib.sha256(bytes.fromhex(e['binary'])).hexdigest(),actions=st['actions'],
        used_blue=any(a['blue'] for a in st['actions']),round_examples=sum(a['rounds'] for a in st['actions']))


def rollout(predict,episodes,batch=16,score=True):
    results=[]
    for start in range(0,len(episodes),batch):
        eps=episodes[start:start+batch]
        states=[dict(code=bytes.fromhex(e['binary']),actions=[],stop=False,status='budget') for e in eps];active=list(range(len(eps)))
        for step in range(9):
            todo=[]
            for i in active:
                try:common.graph(states[i]['code']);todo.append(i)
                except ValueError as exc:states[i]['status']='parse_error';states[i]['error']=str(exc)
            if not todo:break
            # This is the only inference interface: not eps / labels / cases.
            rows=[{'command':eps[i]['command'],'binary':states[i]['code'].hex()} for i in todo]
            outs=predict(rows);active=[]
            for i,a in zip(todo,outs,strict=True):
                st=states[i];a=dict(a,step=step);st['actions'].append(a)
                if a['stop']:st['stop']=True;st['status']='stop';continue
                if step==8:st['status']='budget';continue
                pos=a['position'];value=a['value']
                if not 0<=pos<len(st['code']):st['status']='padded_pointer';continue
                a['before']=st['code'][pos]
                if value==a['before']:st['status']='no_progress';continue
                raw=bytearray(st['code']);raw[pos]=value;st['code']=bytes(raw);active.append(i)
        for e,st in zip(eps,states,strict=True):
            results.append(score_episode(e,st) if score else {'id':e['id'],'final':st['code'].hex(),'stop':st['stop'],'actions':st['actions']})
    return results


def summarize(rs):
    by=defaultdict(list)
    for r in rs:by['all'].append(r);by[r['family']].append(r)
    out={}
    for k,rr in by.items():
        pairs=defaultdict(list)
        for r in rr:
            if r['need_edit'] and r['lane'] is not None:pairs[r['pair_id']].append(r)
        pp=[p for p in pairs.values() if len(p)==2]
        out[k]={'total':len(rr),'edits':sum(r['need_edit'] for r in rr),'strict_edits':sum(r['need_edit'] and r['strict'] for r in rr),
                'noops':sum(not r['need_edit'] for r in rr),'false_edits':sum(r['false_edit'] for r in rr),
                'strict_total':sum(r['strict'] for r in rr),'pairs':len(pp),'pairs_both':sum(all(r['strict'] for r in p) for p in pp),
                'blue_episodes':sum(r['used_blue'] for r in rr),'blue_actions':sum(a['blue'] for r in rr for a in r['actions']),
                'actions':sum(len(r['actions']) for r in rr),'round_examples':sum(r['round_examples'] for r in rr),
                'execution_calls':sum(r['execution_calls'] for r in rr),'all_vm_agree':all(r['vm_agreement'] for r in rr)}
    return out


def compare(base,cand):
    b={r['id']:r for r in base};assert len(base)==len(cand)
    return {'lost_success_ids':[r['id'] for r in cand if b[r['id']]['strict'] and not r['strict']],
            'added_success_ids':[r['id'] for r in cand if not b[r['id']]['strict'] and r['strict']],
            'added_false_edit_ids':[r['id'] for r in cand if not b[r['id']]['false_edit'] and r['false_edit']]}


def calibrate(root,tag):
    configure_torch();m=load_rgb(root/'runs'/tag/'model.pt');out=root/'calibration'/tag;out.mkdir(parents=True,exist_ok=False)
    episodes=read_rows(root/'data/calibration.jsonl');memo=Memo(m);cpu=time.process_time();summaries={};baselines={}
    for mode in ('red_only','blue_only'):
        memo.mode=mode;rs=rollout(memo,episodes);baselines[mode]=rs;summaries[mode]=summarize(rs)
        (out/(mode+'.json')).write_text(json.dumps({'summary':summaries[mode],'records':rs}))
    choices=[]
    thresholds=json.loads((root/'protocol.json').read_text())['calibration']['thresholds']
    for threshold in thresholds:
        memo.mode='auto';memo.threshold=threshold;rs=rollout(memo,episodes);su=summarize(rs);cmp=compare(baselines['red_only'],rs)
        ok=not cmp['lost_success_ids'] and not cmp['added_false_edit_ids'] and su['all']['blue_episodes']>=16
        choices.append({'threshold':threshold,'qualified':ok,'summary':su,'comparison':cmp})
        (out/f'auto-{threshold}.json').write_text(json.dumps({'summary':su,'comparison':cmp,'records':rs}))
    valid=[c for c in choices if c['qualified']]
    best=min(valid,key=lambda c:(c['summary']['all']['round_examples'],c['threshold'])) if valid else None
    selection={'enabled':best is not None,'threshold':best['threshold'] if best else 0.01,
               'model_sha256':file_sha(root/'runs'/tag/'model.pt'),'calibration_sha256':file_sha(root/'data/calibration.jsonl'),
               'choices':choices,'baselines':summaries,'memo_neural_rows':memo.neural_rows,'cpu_seconds':time.process_time()-cpu,
               'population_risk_guarantee':False,'deployment_approved':False}
    # Validate selected policy with actual subset dispatch, not calibration cache.
    m.policy=RoutePolicy(selection['enabled'],selection['threshold']);live=rollout(Live(m,'auto'),episodes)
    cmp=compare(baselines['red_only'],live);su=summarize(live)
    selection['live_selected_summary']=su;selection['live_comparison']=cmp
    (out/'selected-live.json').write_text(json.dumps({'records':live,'summary':su,'comparison':cmp}))
    if cmp['lost_success_ids'] or cmp['added_false_edit_ids']:
        selection['enabled']=False;selection['disabled_reason']='actual subset dispatch failed preservation on calibration'
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    print('CALIBRATED',tag,selection['enabled'],selection['threshold'],su['all'],flush=True)


def final(root,tag):
    configure_torch();freeze=json.loads((root/'final-freeze.json').read_text())
    assert file_sha(root/'runs'/tag/'model.pt')==freeze['models'][tag]
    assert file_sha(root/'data/test.jsonl')==freeze['data']['test.jsonl']
    assert file_sha(Path(__file__))==freeze['evaluator_sha256']
    m=load_rgb(root/'runs'/tag/'model.pt');sel=json.loads((root/'calibration'/tag/'selection.json').read_text())
    assert file_sha(root/'calibration'/tag/'selection.json')==freeze['selections'][tag]
    m.policy=RoutePolicy(sel['enabled'],sel['threshold']);out=root/'final'/tag;out.mkdir(parents=True,exist_ok=False)
    eps=read_rows(root/'data/test.jsonl');results={}
    for mode in ('red_only','blue_only','auto'):
        cpu=time.process_time();rs=rollout(Live(m,mode),eps);summary=summarize(rs);results[mode]=rs
        (out/(mode+'.json')).write_text(json.dumps({'summary':summary,'cpu_seconds':time.process_time()-cpu,'records':rs}))
        print('FINAL',tag,mode,summary['all'],flush=True)
    (out/'comparison.json').write_text(json.dumps(compare(results['red_only'],results['auto']),indent=2))
    # Complete previously public regression; no post-result policy selection.
    eps=read_rows(root/'data/regression.jsonl')
    for mode in ('red_only','auto'):
        rs=rollout(Live(m,mode),eps);results['reg_'+mode]=rs
        (out/('regression-'+mode+'.json')).write_text(json.dumps({'summary':summarize(rs),'records':rs}))
    (out/'regression-comparison.json').write_text(json.dumps(compare(results['reg_red_only'],results['reg_auto']),indent=2))
    print('REGRESSION_DONE',tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibrate','final']);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--tag',required=True);a=p.parse_args()
    (calibrate if a.phase=='calibrate' else final)(a.root.resolve(),a.tag)
