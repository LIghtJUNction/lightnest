"""Re-run blue drafts to check EVERY learned routing decision and record its risk.

Runs only after frozen primary evaluation. No tuning or checkpoint changes.
"""
import json,time
from pathlib import Path
from collections import Counter
import torch
from experiment.run_study import ROOT,read,write,project,load_rgb,load_risk,configure_torch,compact,draft_label
from rgb.sequence import Predictor,drafts,signature

def main():
 configure_torch();allchecks=[];cpu=time.process_time();seqs=0
 for seed in (104729,130363):
  model=load_rgb(ROOT/f'models/rgb-{seed}.pt');risk={arm:load_risk(seed,arm) for arm in ('full','stats')}
  policies={arm:json.loads((ROOT/f'policies/risk-{arm}-{seed}.json').read_text()) for arm in risk}
  for dataset,path in [('normal','data/normal.jsonl'),('stress','data/stress.jsonl'),('regression','baseline_data/regression.jsonl')]:
   es=read(ROOT/path);dlist=[];blue=Predictor(model,'blue_only')
   for i in range(0,len(es),16):dlist.extend(drafts(blue,project(es[i:i+16])))
   seqs+=len(es);X=torch.tensor([d['features'] for d in dlist]);labs=torch.tensor([float(draft_label(e,compact(d))) for e,d in zip(es,dlist)])
   details=[]
   for arm,head in risk.items():
    with torch.no_grad():pr=head(X).sigmoid().tolist()
    result=json.loads((ROOT/f'results/final-{seed}-{dataset}-risk_{arm}.json').read_text());rs=result['records'];pol=policies[arm];cnt=Counter();saved=[]
    for e,d,p,r,label in zip(es,dlist,pr,rs,labs.tolist(),strict=True):
     accept=p<=pol['threshold'] and d['state']['stop'] and d['state']['status']=='stop'
     assert accept==r['used_blue'],(e['id'],arm,'gate decision mismatch')
     if 'draft' in r:
      assert d['state']['code'].hex()==r['draft']['final']
      assert [signature(a) for a in d['state']['actions']]==[signature(a) for a in r['draft']['actions']]
     cnt['accepted']+=bool(accept);cnt['unsafe_accepted']+=bool(accept and label)
     cnt['accepted_edit_tasks']+=bool(accept and e['edits']);cnt['accepted_with_actual_edits']+=bool(accept and any(not a['stop'] for a in d['state']['actions']))
     cnt['wrong_draft_noop_proposals']+=bool(not e['edits'] and any(not a['stop'] for a in d['state']['actions']))
     saved.append({'id':e['id'],'predicted_sequence_error':p,'accepted':bool(accept),'draft_unsafe_label':bool(label),'blue_actions':len(d['state']['actions'])})
    details.append({'seed':seed,'dataset':dataset,'arm':arm,'counts':dict(cnt),'brier':sum((p-y)**2 for p,y in zip(pr,labs.tolist()))/len(es),'decisions':saved})
   allchecks.extend(details)
 write(ROOT/'results/risk-decision-replay.json',{'passed':True,'new_blue_sequences':seqs,'checked_gate_decisions':2*seqs,'records':allchecks,'cpu_seconds':time.process_time()-cpu,'after_final_diagnostic_only':True})
 print('RISK_DECISION_REPLAY',seqs,flush=True)
if __name__=='__main__':main()
