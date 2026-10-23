"""Post-primary backend experiment. No parameter, route or quality-rule changes."""
import json,time,datetime,statistics
from pathlib import Path
from collections import defaultdict
import torch
from rgb.runtime import common
from rgb.scripted_core import load_compiled_rgb
from experiment.run_study import ROOT,read,write,sha,run_mode,load_rgb,configure_torch,project,signature

def validate(seed):
 configure_torch();protocol=json.loads((ROOT/'backend-protocol.json').read_text())
 for n,h in protocol['source'].items():assert sha(ROOT/n)==h
 a=load_rgb(ROOT/f'models/rgb-{seed}.pt');b=load_compiled_rgb(ROOT/f'models/rgb-{seed}.pt');tensorchecks=[];checks=[]
 by=defaultdict(list)
 for r in read(ROOT/'baseline_data/calibration.jsonl'):by[r['family']].append(r)
 for fam in sorted(by):
  rows=project(by[fam][:8]);enc=common.encode(rows,a.red.config)
  x=a(*enc,mode='red_only');y=b(*enc,mode='red_only')
  eq=all(torch.equal(x[k],y[k]) for k in ('scores','bits'));assert eq,(seed,fam)
  tensorchecks.append({'family':fam,'rows':len(rows),'scores_and_bits_bitwise_equal':True})
 for data,path in [('normal','data/normal.jsonl'),('stress','data/stress.jsonl'),('regression','baseline_data/regression.jsonl')]:
  eps=read(ROOT/path)
  for mode in ('red','risk_full'):
   ref=json.loads((ROOT/f'results/final-{seed}-{data}-{mode}.json').read_text())['records'];o=run_mode(b,eps,seed,mode,score=False);changed=[]
   for x,y in zip(ref,o['records'],strict=True):
    if x['id']!=y['id'] or x['final']!=y['final'] or x['status']!=y['status'] or [signature(z) for z in x['actions']]!=[signature(z) for z in y['actions']]:changed.append(x['id'])
   assert not changed,(seed,data,mode,changed[:5]);checks.append({'data':data,'mode':mode,'rows':len(ref),'all_discrete_actions_equal':True,'cpu_seconds':o['cpu_seconds']})
   write(ROOT/f'results/backend-{seed}-{data}-{mode}.json',o)
   print('BACKEND_EQ',seed,data,mode,len(ref),flush=True)
 write(ROOT/f'results/backend-validation-{seed}.json',{'passed':True,'tensor_checks':tensorchecks,'checks':checks,'setup':b.backend_setup,
  'same_registered_state':all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()),'scope':'same already-consumed cases, implementation equivalence, not a new generalization result'})

def benchmark():
 configure_torch();by=defaultdict(list)
 for r in read(ROOT/'baseline_data/calibration.jsonl'):by[r['family']].append(r)
 works={'simple':by['simple'][:32],'balanced':[r for fam in sorted(by) for r in by[fam][:4]]};result=[];setup=[]
 for seed in (104729,130363):
  a=load_rgb(ROOT/f'models/rgb-{seed}.pt');b=load_compiled_rgb(ROOT/f'models/rgb-{seed}.pt');setup.append({'seed':seed}|b.backend_setup)
  for workload,eps in works.items():
   for batch in (1,16):
    variants=[('eager_red',a,'red'),('compiled_red',b,'red'),('compiled_risk',b,'risk_full')];ts=defaultdict(list)
    for tag,m,mode in variants:run_mode(m,eps,seed,mode,batch=batch,score=False)
    for repeat in range(5):
     v=variants[repeat%3:]+variants[:repeat%3]
     for tag,m,mode in v:
      o=run_mode(m,eps,seed,mode,batch=batch,score=False);ts[tag].append(o['wall_seconds'])
    ref=statistics.median(ts['eager_red'])
    for tag,_,_ in variants:result.append({'seed':seed,'workload':workload,'batch':batch,'variant':tag,'seconds':ts[tag],'median':statistics.median(ts[tag]),'speedup_vs_eager_red':ref/statistics.median(ts[tag])})
    print('BACKEND_TIME',seed,workload,batch,{k:round(ref/statistics.median(v),3) for k,v in ts.items()},flush=True)
 write(ROOT/'results/backend-timing.json',{'records':result,'setup':setup,'warm_steady_state_only':True,'repeats':5,'cache':False})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['validate','benchmark']);p.add_argument('--seed',type=int);a=p.parse_args()
 validate(a.seed) if a.phase=='validate' else benchmark()
