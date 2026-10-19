"""Small TRAIN-only cache equivalence and throughput check before training."""
import json,time,torch
from pathlib import Path
from rgb.network import RGBNetwork
from rgb.runtime import configure_torch,common
from .cache import read_rows,collate
from .train import loss_actor,Sampler
configure_torch();torch.manual_seed(70001)
rows=read_rows('data/fit.jsonl');cache=torch.load('caches/parent-104729.pt',weights_only=True,map_location='cpu')['entries']
m=RGBNetwork(common.load_parent(104729)).train();batch=Sampler(rows,70001).sample()
with torch.no_grad():
 ctx,h,_,_=collate(cache,batch,4);_,_,live_ctx,live=m.encode_shared(*common.encode(batch,m.red.config))
 error=float((h-live).abs().max());assert torch.allclose(h,live,atol=2e-5,rtol=2e-5)
 a=m.fast_readout(ctx,h)[1];b=m.fast_readout(live_ctx,live)[1]
 assert torch.equal(a[0].argmax(-1),b[0].argmax(-1))
opt=torch.optim.AdamW(list(m.green.parameters())+list(m.blue.parameters()),lr=3e-4);cpu=time.process_time()
for _ in range(3):
 opt.zero_grad(set_to_none=True);loss,_=loss_actor(m,cache,batch);loss.backward();opt.step()
Path('results/preflight.json').write_text(json.dumps({'max_cached_state_error':error,'same_next_positions':True,'pilot_updates':3,'pilot_training_seconds':time.process_time()-cpu,'pilot_model_discarded':True,'test_requests':0},indent=2))
print(Path('results/preflight.json').read_text())
