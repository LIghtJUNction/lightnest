"""Post-evaluation integrity, sampled-position and save/restore checks."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import json,torch,subprocess,sys,hashlib
from . import ROOT,BASE
from .model import *
from .prepare import read,write,sha
from lightnest_data.vm import execute_b

def main():
 configure_torch();rows=read(ROOT/'data/mode_train.jsonl');second_calls=0;rollout_checks=0;sampled_checks=0;restores=[]
 for seed in (104729,130363):
  for mode in ('rl0','rlcost'):
   d=json.loads((ROOT/f'results/rl-{mode}-{seed}.json').read_text());cost=0. if mode=='rl0' else .1
   for row in d['rollouts']:
    e=rows[row['train_index']];r=row['record'];b=bytearray.fromhex(e['binary']);edits=0;stopped=False;decisions=[];rounds=0
    for j,a in enumerate(r['actions']):
     rounds+=a['rounds'];decisions += [int(t['mode']=='EMIT') for t in a['mode_trace'] if t['depth']<64]
     if a['stop']:stopped=True;break
     edits+=1
     if j>=8 or not 0<=a['position']<len(b):break
     if b[a['position']]==a['value']:break
     b[a['position']]=a['value']
    assert decisions==[s['choice'] for s in row['sampled_modes']]
    sampled_checks+=len(decisions)
    assert bytes(b).hex()==r['final']
    ok=True
    for x,y in e['cases']:
     try:v=execute_b(bytes(b),x)
     except (ValueError,IndexError):v=None
     second_calls+=1
     if v!=y:ok=False
    strict=bool(bytes(b).hex()==e['target'] and stopped and ok and (e['edits'] or edits==0))
    bad=not e['edits'] and edits>0
    reward=(1 if strict else -1)-(1 if bad else 0)-cost*rounds/(9*64)
    assert strict==r['strict'] and abs(reward-row['reward'])<1e-12
    rollout_checks+=1
  for mode in ('sft','rl0','rlcost'):
   saved=torch.load(ROOT/f'models/policy-{mode}-{seed}.pt',weights_only=True,map_location='cpu')
   copies=[];optimizers=[]
   for _ in range(2):
    p=ModePolicy();p.load_state_dict(deepcopy(saved['policy']));opt=torch.optim.AdamW(p.parameters());opt.load_state_dict(deepcopy(saved['optimizer']));copies.append(p);optimizers.append(opt)
   gen=torch.Generator().manual_seed(31003);x=torch.randn(16,FEATURES,generator=gen);y=torch.randint(2,(16,),generator=gen)
   for p,opt in zip(copies,optimizers):
    opt.zero_grad();loss=torch.nn.functional.cross_entropy(p(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step()
   assert all(torch.equal(v,copies[1].state_dict()[k]) for k,v in copies[0].state_dict().items())
   restores.append({'seed':seed,'mode':mode,'same_next_update_after_restore':True,'diagnostic_updates':2})
 # CLI only receives command + bytes. Check both neural branches explicitly.
 e=read(BASE/'baseline_data/calibration.jsonl')[0];src=ROOT/'results/cli-source.rv32';src.write_bytes(bytes.fromhex(e['binary']));before=sha(src);cli=[]
 for seed in (104729,130363):
  for mode in ('parent64','rl0'):
   out=ROOT/f'results/cli-{seed}-{mode}'
   p=subprocess.run([sys.executable,'-m','hybrid.predict','--seed',str(seed),'--strategy',mode,'--binary',str(src),'--command',e['command'],'--out',str(out)],capture_output=True,text=True,check=True,cwd=ROOT)
   assert sha(src)==before;assert (out/'candidate.rv32').stat().st_mode&0o777==0o600
   cli.append(json.loads(p.stdout))
 rejected=subprocess.run([sys.executable,'-m','hybrid.predict','--seed','104729','--strategy','parent64','--binary',str(src),'--command',e['command'],'--out',str(ROOT/'results/cli-104729-parent64')],capture_output=True,text=True,cwd=ROOT)
 assert rejected.returncode!=0
 write(ROOT/'results/final-checks.json',{'passed':True,'rl_rollouts_recomputed':rollout_checks,'rl_sampled_positions_checked':sampled_checks,
      'second_vm_calls':second_calls,'restores':restores,'cli_runs':cli,'repeated_output_rejected':True,'input_unchanged':True,
      'diagnostic_update_count':len(restores)*2,'host_binary_execution':False})
 print('FINAL_CHECKS',rollout_checks,sampled_checks,second_calls,flush=True)
if __name__=='__main__':main()
