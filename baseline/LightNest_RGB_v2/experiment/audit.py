"""Separate count/replay code. Does not use study.evaluate scoring functions."""
import json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import torch
from torch.nn import functional as F
from rgb.sequence import SequenceRisk
from rgb.runtime import configure_torch
from lightnest_data.vm import execute_b
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)

def train():
 configure_torch();cfg=json.loads((ROOT/'protocol.json').read_text());checks=[];total=0
 for seed in (104729,130363):
  assert sha(ROOT/f'models/rgb-{seed}.pt')==cfg['models'][str(seed)]
  rows=read(ROOT/'baseline_data/router_fit.jsonl');pred=json.loads((ROOT/f'results/fit-drafts-{seed}.json').read_text())['records']
  y=[]
  for e,d in zip(rows,pred,strict=True):
   assert e['id']==d['id'];v=d['draft'];label=not (v['final']==e['target'] and v['stop'] and (e['edits'] or all(a['stop'] for a in v['actions'])))
   assert bool(label)==d['unsafe_label'];y.append(float(label))
  x=torch.tensor([p['draft']['features'] for p in pred]);y=torch.tensor(y)
  for arm in ('full','stats'):
   path=ROOT/f'models/risk-{arm}-{seed}.pt';d=torch.load(path,map_location='cpu',weights_only=True);log=json.loads((ROOT/f'results/fit-{arm}-{seed}.json').read_text())
   assert d['steps']==512 and d['protocol_sha256']==sha(ROOT/'protocol.json')
   assert d['rgb_sha256']==sha(ROOT/f'models/rgb-{seed}.pt');assert d['fit_sha256']==sha(ROOT/'baseline_data/router_fit.jsonl')
   g=torch.Generator().manual_seed(seed+9981);stream=[torch.randint(len(rows),(32,),generator=g).tolist() for _ in range(512)]
   assert stream==log['sample_indices'];assert torch.equal(g.get_state(),d['generator']);total+=len(stream)*32
   for v in d['state_dict'].values():assert bool(torch.isfinite(v).all())
   for st in d['optimizer']['state'].values():assert int(st['step'])==512
   # Two independent restores perform the same next sampled training step.
   after=[]
   for rep in range(2):
    dd=torch.load(path,map_location='cpu',weights_only=True);m=SequenceRisk(dd['stats_only']);m.load_state_dict(dd['state_dict'])
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);opt.load_state_dict(dd['optimizer']);g=torch.Generator();g.set_state(dd['generator'])
    ids=torch.randint(len(rows),(32,),generator=g);opt.zero_grad(set_to_none=True)
    loss=F.binary_cross_entropy_with_logits(m(x[ids]),y[ids],pos_weight=torch.tensor(4.));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    after.append(({k:v.clone() for k,v in m.state_dict().items()},opt.state_dict(),ids))
   assert torch.equal(after[0][2],after[1][2])
   assert all(torch.equal(after[0][0][k],after[1][0][k]) for k in after[0][0])
   for key,st in after[0][1]['state'].items():
    for k,v in st.items():
     w=after[1][1]['state'][key][k];assert torch.equal(v,w) if torch.is_tensor(v) else v==w
   checks.append({'seed':seed,'arm':arm,'stream_reconstructed':True,'resume_next_step_equal':True,'parameters':sum(p.numel() for p in m.parameters())})
 save(ROOT/'results/training-audit-v2.json',{'passed':True,'checks':checks,'presentations_reconstructed':total,'base_rgb_unchanged':True,'heads_trained':4,'formal_updates':2048,'extra_resume_steps':8})
 print('TRAIN_AUDIT',total,flush=True)

def replay(e,rec):
 raw=bytearray.fromhex(e['binary']);stop=False;proposals=0
 for i,a in enumerate(rec['actions']):
  assert a['step']==i and i<9
  if a['stop']:stop=True;assert i==len(rec['actions'])-1;break
  proposals+=1
  if i==8:continue
  p=a['position']
  if not 0<=p<len(raw):continue
  assert a['before']==raw[p];raw[p]=a['value']
 assert raw.hex()==rec['final'];ok=True;calls=0
 for x,y in e['cases']:
  calls+=1
  try:v=execute_b(bytes(raw),x)
  except (ValueError,IndexError):v=None
  ok &= v==y
 need=bool(e['edits']);strict=bool(raw.hex()==e['target'] and stop and ok and (need or proposals==0));false=not need and proposals>0
 assert strict==rec['strict'] and bool(ok)==rec['numeric'] and false==rec['false_edit']
 assert stop==rec['explicit_stop'] and proposals==rec['proposed']
 if 'draft' in rec:
  d=rec['draft'];dr=bytearray.fromhex(e['binary'])
  for i,a in enumerate(d['actions']):
   assert a['step']==i
   if a['stop']:break
   if i==8:continue
   p=a['position']
   if not 0<=p<len(dr):continue
   assert a['before']==dr[p];dr[p]=a['value']
  assert dr.hex()==d['final']
  assert rec['draft_edit_proposals']==sum(not a['stop'] for a in d['actions'])
 return calls,strict,false

def final():
 start=time.process_time();calls=0;count=0;checks=[];data={k:read(ROOT/p) for k,p in [('normal','data/normal.jsonl'),('stress','data/stress.jsonl'),('regression','baseline_data/regression.jsonl')]}
 freeze=json.loads((ROOT/'final-freeze.json').read_text())
 for name,h in freeze['files'].items():assert sha(ROOT/name)==h,name
 for seed in (104729,130363):
  for dataset,eps in data.items():
   by={r['id']:r for r in eps};baseline=None
   for mode in ('red','v1_stop','risk_stats','risk_full','verified'):
    obj=json.loads((ROOT/f'results/final-{seed}-{dataset}-{mode}.json').read_text());rs=obj['records'];assert len(rs)==len(by)==len({r['id'] for r in rs})
    ct=Counter();pairs=defaultdict(list)
    for r in rs:
     e=by[r['id']];n,strict,false=replay(e,r);calls+=n;count+=1;need=bool(e['edits'])
     ct['total']+=1;ct['edits']+=need;ct['noops']+=not need;ct['strict_total']+=strict;ct['strict_edits']+=need and strict;ct['false_edits']+=false
     if need and e.get('lane') is not None:pairs[e['pair_id']].append(strict)
    pp=[p for p in pairs.values() if len(p)==2];ct['pairs']=len(pp);ct['pairs_both']=sum(all(p) for p in pp)
    for k,v in ct.items():assert obj['summary']['all'][k]==v,(mode,dataset,k)
    if mode=='red':baseline={r['id']:r for r in rs}
    else:
     cmp={'lost_success_ids':[],'added_success_ids':[],'added_false_edit_ids':[]};diff=[]
     for r in rs:
      b=baseline[r['id']]
      if b['strict'] and not r['strict']:cmp['lost_success_ids'].append(r['id'])
      if not b['strict'] and r['strict']:cmp['added_success_ids'].append(r['id'])
      if not b['false_edit'] and r['false_edit']:cmp['added_false_edit_ids'].append(r['id'])
      sig=lambda rr:[(a['stop'],a['position'],a['value']) for a in rr['actions']]
      if r['final']!=b['final'] or r['status']!=b['status'] or sig(r)!=sig(b):diff.append(r['id'])
     assert cmp==obj['comparison'];assert diff==obj['equivalence']['different_discrete_trajectory_ids']
     if mode=='verified':assert not diff,'verified action preservation failed'
    checks.append({'seed':seed,'data':dataset,'mode':mode,'rows':len(rs),'metric_replay_passed':True})
 save(ROOT/'results/evaluation-audit-v2.json',{'passed':True,'sequences':count,'second_vm_executions':calls,'checks':checks,'cpu_seconds':time.process_time()-start,'scope':'same-session separate count/replay code, supplied second interpreter, not external audit'})
 print('FINAL_AUDIT',count,calls,flush=True)
if __name__=='__main__':
 import sys
 train() if sys.argv[1]=='train' else final()
