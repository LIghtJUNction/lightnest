"""Fit full-sequence risk, calibrate on public cases, then evaluate a sealed holdout.

Every phase writes to a new file; baseline RGB checkpoints remain unchanged.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,random,time,resource
from pathlib import Path
from collections import Counter
import torch
from torch.nn import functional as F
from rgb.runtime import configure_torch
from rgb.network import load_rgb
from rgb.stop_gate import load_stop_only,StopOnlyPolicy
from rgb.sequence import (FEATURES,SequenceRisk,Predictor,drafts,new_state,continue_red,
                         verified_drafts,risk_drafts,signature)
from study.evaluate import rollout,Live,score_episode,summarize,compare

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def write(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False)
def compact(d):
    st=d['state']
    return {'final':st['code'].hex(),'stop':st['stop'],'status':st['status'],'actions':st['actions'],
        'features':d['features']}
def raw_state(x):
    return {'code':bytes.fromhex(x['final']),'stop':x['stop'],'status':x['status'],'actions':x['actions']}
def draft_label(e,x):
    return not (x['final']==e['target'] and x['stop'] and (bool(e['edits']) or all(a['stop'] for a in x['actions'])))
def project(rows):return [{'command':r['command'],'binary':r['binary']} for r in rows]

def fit(seed):
    configure_torch();m=load_rgb(ROOT/f'models/rgb-{seed}.pt');rows=read(ROOT/'baseline_data/router_fit.jsonl');blue=Predictor(m,'blue_only')
    cpu=time.process_time();xs=[]
    for start in range(0,len(rows),16):xs.extend(compact(d) for d in drafts(blue,project(rows[start:start+16])))
    cachecpu=time.process_time()-cpu
    write(ROOT/f'results/fit-drafts-{seed}.json',{'records':[{'id':r['id'],'draft':x,'unsafe_label':draft_label(r,x)} for r,x in zip(rows,xs,strict=True)],'cpu_seconds':cachecpu,'blue_rows':blue.rows,'calls':blue.calls})
    X=torch.tensor([x['features'] for x in xs]);Y=torch.tensor([draft_label(r,x) for r,x in zip(rows,xs,strict=True)],dtype=torch.float32)
    cfg=json.loads((ROOT/'protocol.json').read_text())['risk_training'];rec={}
    for arm in ('full','stats'):
        torch.manual_seed(seed);risk=SequenceRisk(stats_only=arm=='stats');risk.normalize_from(X)
        init={k:v.clone() for k,v in risk.state_dict().items()};opt=torch.optim.AdamW(risk.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        gen=torch.Generator().manual_seed(seed+9981);stream=[];losses=[];start=time.process_time()
        for step in range(cfg['steps']):
            ids=torch.randint(len(rows),(cfg['batch'],),generator=gen);stream.append(ids.tolist());opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(risk(X[ids]),Y[ids],pos_weight=torch.tensor(float(cfg['unsafe_weight'])))
            loss.backward();torch.nn.utils.clip_grad_norm_(risk.parameters(),cfg['clip_norm']);opt.step();losses.append(float(loss.detach()))
        cpu=time.process_time()-start;risk.eval();changed=sum(int((init[k]!=v).sum()) for k,v in risk.state_dict().items())
        payload={'architecture':'rgb-v2-sequence-risk-v1','stats_only':arm=='stats','state_dict':risk.state_dict(),
                 'optimizer':opt.state_dict(),'generator':gen.get_state(),'torch_rng':torch.get_rng_state(),'steps':cfg['steps'],
                 'rgb_sha256':sha(ROOT/f'models/rgb-{seed}.pt'),'fit_sha256':sha(ROOT/'baseline_data/router_fit.jsonl'),'protocol_sha256':sha(ROOT/'protocol.json')}
        p=ROOT/f'models/risk-{arm}-{seed}.pt'
        with p.open('xb') as f:torch.save(payload,f)
        rec[arm]={'training_cpu_seconds':cpu,'parameters':sum(p.numel() for p in risk.parameters()),'changed_values':changed,
                   'loss_first':losses[0],'loss_last':losses[-1],'model_sha256':sha(p),'steps':cfg['steps'],
                   'presentations':cfg['steps']*cfg['batch'],'unique_rows':len({i for z in stream for i in z}),
                   'stream_sha256':hashlib.sha256(json.dumps(stream).encode()).hexdigest()}
        write(ROOT/f'results/fit-{arm}-{seed}.json',{'summary':rec[arm],'row_ids':[r['id'] for r in rows],'sample_indices':stream,'losses':losses})
    assert rec['full']['stream_sha256']==rec['stats']['stream_sha256']
    write(ROOT/f'results/training-{seed}.json',{'arms':rec,'unsafe_fit_rows':int(Y.sum()),'total_fit_rows':len(rows),'draft_cpu_seconds':cachecpu,'base_sha256':sha(ROOT/f'models/rgb-{seed}.pt')})
    print('FIT',seed,json.dumps(rec),flush=True)

def load_risk(seed,arm):
    d=torch.load(ROOT/f'models/risk-{arm}-{seed}.pt',map_location='cpu',weights_only=True)
    if d['rgb_sha256']!=sha(ROOT/f'models/rgb-{seed}.pt'):raise ValueError('risk/base binding mismatch')
    if any(not bool(torch.isfinite(v).all()) for v in d['state_dict'].values()):raise ValueError('nonfinite risk')
    r=SequenceRisk(d['stats_only']);r.load_state_dict(d['state_dict']);return r.eval()

def score_with_meta(e,st,d=None):
    x=score_episode(e,st)
    x['draft_edit_proposals']=sum(not a['stop'] for a in d['state']['actions']) if d else 0
    x['draft_noop_edit_proposals']=bool(not e['edits'] and x['draft_edit_proposals'])
    if d:x['draft']={'status':d['state']['status'],'stop':d['state']['stop'],'final':d['state']['code'].hex(),'actions':d['state']['actions']}
    return x

def calibrate(seed):
    configure_torch();m=load_rgb(ROOT/f'models/rgb-{seed}.pt');eps=read(ROOT/'baseline_data/calibration.jsonl')+read(ROOT/'baseline_data/regression.jsonl')
    blue=Predictor(m,'blue_only');red=Predictor(m,'red_only');ds=[];redrs=[];bluers=[];cpu=time.process_time()
    for st in range(0,len(eps),16):
        part=eps[st:st+16];rows=project(part);dd=drafts(blue,rows);states=continue_red(red,rows,[new_state(r) for r in rows])
        for e,d,s in zip(part,dd,states,strict=True):
            ds.append(compact(d));redrs.append(score_episode(e,s));bluers.append(score_episode(e,d['state']))
    write(ROOT/f'results/calibration-{seed}.json',{'red':redrs,'blue':bluers,'features':[d['features'] for d in ds],
        'cpu_seconds':time.process_time()-cpu,'red_rows':red.rows,'blue_rows':blue.rows})
    cfg=json.loads((ROOT/'protocol.json').read_text())['calibration'];X=torch.tensor([d['features'] for d in ds]);choices={}
    for arm in ('full','stats'):
        risk=load_risk(seed,arm)
        with torch.no_grad():probs=risk(X).sigmoid().tolist()
        grid=[]
        for t in cfg['thresholds']:
            accepts=[bool(p<=t and d['stop'] and d['status']=='stop') for p,d in zip(probs,ds,strict=True)]
            mixed=[b if a else r for a,b,r in zip(accepts,bluers,redrs,strict=True)];cmp=compare(redrs,mixed)
            ok=not cmp['lost_success_ids'] and not cmp['added_false_edit_ids'] and sum(accepts)>=16
            grid.append({'threshold':t,'qualified':ok,'accepted':sum(accepts),'unsafe_accepted':sum(a and not b['strict'] for a,b in zip(accepts,bluers)),
                         'lost_red_successes':len(cmp['lost_success_ids']),'new_false_edits':len(cmp['added_false_edit_ids']),
                         'strict_total':sum(r['strict'] for r in mixed)})
        valid=[g for g in grid if g['qualified']];selected=min(valid,key=lambda g:(-g['accepted'],g['threshold'])) if valid else None
        sel={'seed':seed,'arm':arm,'enabled':selected is not None,'threshold':selected['threshold'] if selected else -1.,'grid':grid,
             'model_sha256':sha(ROOT/f'models/risk-{arm}-{seed}.pt'),'rgb_sha256':sha(ROOT/f'models/rgb-{seed}.pt'),
             'data_sha256':{n:sha(ROOT/'baseline_data'/n) for n in ('calibration.jsonl','regression.jsonl')},
             'no_population_risk_guarantee':True,'deployment_approved':False}
        write(ROOT/f'policies/risk-{arm}-{seed}.json',sel);choices[arm]=sel
    print('CAL',seed,{a:(s['enabled'],s['threshold']) for a,s in choices.items()},flush=True)

def equivalence(red,cand):
    r={x['id']:x for x in red};different=[]
    for c in cand:
        b=r[c['id']]
        if c['final']!=b['final'] or c['status']!=b['status'] or [signature(x) for x in c['actions']]!=[signature(x) for x in b['actions']]:different.append(c['id'])
    return {'different_discrete_trajectory_ids':different,'matched':len(cand)-len(different),'total':len(cand)}

def run_mode(m,eps,seed,mode,batch=16,score=True):
    blue=Predictor(m,'blue_only');red=Predictor(m,'red_only');out=[];met=Counter();cpu=time.process_time();wall=time.perf_counter()
    if mode=='v1_stop':
        old=load_stop_only(ROOT/f'models/rgb-{seed}.pt');p=json.loads((ROOT/f'policies/v1-stop-{seed}.json').read_text());old.policy=StopOnlyPolicy(p['enabled'],p['threshold'])
        # No checkpoint load inside timed section when called for this mode.
        wall=time.perf_counter();cpu=time.process_time();out=rollout(Live(old,'auto'),eps,batch=batch,score=score)
        return {'records':out,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'compute':{},'summary':summarize(out) if score else {}}
    risk=None;threshold=-1
    if mode.startswith('risk_'):
        arm=mode.split('_')[1];risk=load_risk(seed,arm);pol=json.loads((ROOT/f'policies/risk-{arm}-{seed}.json').read_text());threshold=pol['threshold']
        if pol['model_sha256']!=sha(ROOT/f'models/risk-{arm}-{seed}.pt'):raise ValueError('policy binding mismatch')
    wall=time.perf_counter();cpu=time.process_time()
    for start in range(0,len(eps),batch):
        es=eps[start:start+batch];rows=project(es);ds=None
        if mode=='red' or (mode.startswith('risk_') and threshold<0):
            states=continue_red(red,rows,[new_state(r) for r in rows])
        elif mode=='verified':
            states,ds,meta=verified_drafts(blue,red,rows,draft_steps=4,verify_batch=32);met.update(meta)
        elif mode.startswith('risk_'):
            states,ds,meta=risk_drafts(blue,red,risk,rows,threshold);met['accepted_drafts']+=meta['accepted_drafts']
        elif mode=='blue':
            ds=drafts(blue,rows);states=[d['state'] for d in ds]
        else:raise ValueError(mode)
        for j,(e,st) in enumerate(zip(es,states,strict=True)):
            d=ds[j] if ds else None
            x=score_with_meta(e,st,d) if score else {'id':e['id'],'final':st['code'].hex(),'status':st['status'],'actions':st['actions'],'stop':st['stop']}
            out.append(x)
    met.update(blue_rows=blue.rows,blue_calls=blue.calls,red_rows=red.rows,red_calls=red.calls)
    return {'records':out,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'compute':dict(met),'summary':summarize(out) if score else {}}

def final(seed):
    configure_torch();freeze=json.loads((ROOT/'final-freeze.json').read_text())
    for name,h in freeze['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('freeze mismatch '+name)
    m=load_rgb(ROOT/f'models/rgb-{seed}.pt')
    for dataset,path in [('normal','data/normal.jsonl'),('stress','data/stress.jsonl'),('regression','baseline_data/regression.jsonl')]:
        eps=read(ROOT/path);red=None
        for mode in ('red','v1_stop','risk_stats','risk_full','verified'):
            obj=run_mode(m,eps,seed,mode);obj.update(seed=seed,dataset=dataset,mode=mode)
            if red is None:red=obj['records']
            else:obj['comparison']=compare(red,obj['records']);obj['equivalence']=equivalence(red,obj['records'])
            write(ROOT/f'results/final-{seed}-{dataset}-{mode}.json',obj)
            print('FINAL',seed,dataset,mode,obj['summary']['all']['strict_total'],len(obj.get('comparison',{}).get('lost_success_ids',[])),round(obj['cpu_seconds'],3),flush=True)
    print('FINAL_DONE',seed,flush=True)

def freeze():
    names=['protocol.json','data/normal.jsonl','data/stress.jsonl','baseline_data/calibration.jsonl','baseline_data/regression.jsonl']
    names +=[str(p.relative_to(ROOT)) for p in (ROOT/'models').glob('*.pt')]
    names +=[str(p.relative_to(ROOT)) for p in (ROOT/'policies').glob('*.json')]
    names +=[str(p.relative_to(ROOT)) for folder in ('rgb','experiment','runtime','lightnest_data','study') for p in (ROOT/folder).rglob('*.py')]
    write(ROOT/'final-freeze.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{n:sha(ROOT/n) for n in names},'holdout_model_queries_so_far':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','calibrate','freeze','final']);p.add_argument('--seed',type=int,default=104729);a=p.parse_args()
    globals()[a.phase]() if a.phase=='freeze' else globals()[a.phase](a.seed)
