"""Assemble every model result; do not pick a seed or retune a policy."""
import json,hashlib,platform,sys
from pathlib import Path
from collections import Counter
import torch
from experiment.run_study import ROOT,read,write,compare,equivalence

def main():
 results={};comparisons={};combined={}
 for seed in (104729,130363):
  s=str(seed);results[s]={};comparisons[s]={};combined[s]={}
  for data in ('normal','stress','regression'):
   objs={m:json.loads((ROOT/f'results/final-{seed}-{data}-{m}.json').read_text()) for m in ('red','v1_stop','risk_stats','risk_full','verified')}
   results[s][data]={m:{k:v for k,v in o.items() if k!='records'} for m,o in objs.items()}
   comparisons[s][data]={m:compare(objs['v1_stop']['records'],o['records']) for m,o in objs.items() if m!='v1_stop'}
  for m in ('red','v1_stop','risk_stats','risk_full','verified'):
   merged={k:results[s]['normal'][m]['summary']['all'][k]+results[s]['stress'][m]['summary']['all'][k] for k in ('total','edits','noops','strict_total','strict_edits','false_edits','pairs','pairs_both')}
   losses=[];added=[];vs_v1=[]
   for ds in ('normal','stress'):
    cmp=results[s][ds][m].get('comparison',{})
    losses+=cmp.get('lost_success_ids',[]);added+=cmp.get('added_success_ids',[])
    vs_v1+=comparisons[s][ds].get(m,{}).get('lost_success_ids',[])
   combined[s][m]=merged|{'lost_vs_red':len(losses),'added_vs_red':len(added),'lost_vs_v1_stop':len(vs_v1)}
 diag=json.loads((ROOT/'results/risk-decision-replay.json').read_text())
 out={'experiment':'LightNest_RGB_v2','combined_new_tests':combined,'per_dataset':results,'comparisons_against_v1_stop':comparisons,
    'risk_decision_summary':[{k:v for k,v in x.items() if k!='decisions'} for x in diag['records']],
    'training':[json.loads((ROOT/f'results/training-{seed}.json').read_text()) for seed in (104729,130363)],
    'audit':json.loads((ROOT/'results/evaluation-audit-v2.json').read_text()),
    'training_audit':json.loads((ROOT/'results/training-audit-v2.json').read_text()),
    'timing':json.loads((ROOT/'results/timing-v2.json').read_text()),
    'new_data':json.loads((ROOT/'results/construction.json').read_text()),
    'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'platform':platform.platform()},
    'stable_parent_replaced':False,'default_auto_enabled':False}
 if (ROOT/'results/backend-timing.json').exists():
  out['backend_timing']=json.loads((ROOT/'results/backend-timing.json').read_text())
  out['backend_validation']=[json.loads((ROOT/f'results/backend-validation-{s}.json').read_text()) for s in (104729,130363)]
 write(ROOT/'results/summary-v2.json',out)
 print(json.dumps(combined,indent=2))
if __name__=='__main__':main()
