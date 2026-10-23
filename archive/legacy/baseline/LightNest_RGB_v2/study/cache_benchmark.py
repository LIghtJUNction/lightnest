"""TRAIN-only diagnostic: cached vs uncached, same frozen core/objective.

Use fresh throwaway models. Never alter the six formal endpoints. Amortization
must include one-time feature-cache build costs; not an inference speed claim.
"""
from copy import deepcopy
import json,time,torch
from pathlib import Path
from torch.nn import functional as F
from rgb.runtime import common,configure_torch
from rgb.network import RGBNetwork
from rgb.losses import action_correct
from .cache import read_rows
from .train import Sampler,loss_actor
configure_torch();torch.manual_seed(23498)
rows=read_rows('data/fit.jsonl');sampler=Sampler(rows,700004);batches=[sampler.sample() for _ in range(32)]
cache=torch.load('caches/parent-104729.pt',weights_only=True,map_location='cpu')['entries']
parent=RGBNetwork(common.load_parent(104729)).train();results=[];final={}
for cached in (False,True):
 m=deepcopy(parent);opt=torch.optim.AdamW(list(m.green.parameters())+list(m.blue.parameters()),lr=3e-4,weight_decay=.01);c=time.process_time();w=time.perf_counter();losses=[]
 for b in batches:
  opt.zero_grad(set_to_none=True)
  if cached:loss,_=loss_actor(m,cache,b)
  else:
   o=m.forward_train(*common.encode(b,m.red.config));s,bi=o['blue'];rs,rb=o['red'];sup,_,_=common.supervised_loss(s,bi,b,m.none_index)
   ok=action_correct(rs,rb,b,m.none_index)
   kl=F.kl_div(s[ok].log_softmax(-1),rs[ok].softmax(-1),reduction='batchmean') if bool(ok.any()) else sup*0
   loss=sup+.1*kl
  loss.backward();torch.nn.utils.clip_grad_norm_(list(m.green.parameters())+list(m.blue.parameters()),1.);opt.step();losses.append(float(loss.detach()))
 results.append({'cached':cached,'updates':32,'cpu_seconds':time.process_time()-c,'wall_seconds':time.perf_counter()-w,'losses':losses});final[cached]={k:v.clone() for k,v in m.state_dict().items()}
err=max(float((v-final[True][k]).abs().max()) for k,v in final[False].items())
r={'trials':results,'max_weight_difference':err,'exact_weights_equal':all(torch.equal(v,final[True][k]) for k,v in final[False].items()),
   'cpu_speedup_training_only':results[0]['cpu_seconds']/results[1]['cpu_seconds'],
   'one_time_cache_build_seconds':json.loads(Path('caches/parent-104729.json').read_text())['cpu_seconds'],
   'formal_endpoints_modified':False,'scope':'32 update pilot, source dataset TRAIN only; same samples and losses; raw core-forward cost included only in uncached mode'}
Path('results/cache-training-benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='trials'},indent=2))
