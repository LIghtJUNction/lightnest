"""Resume an exact actor-training checkpoint into a NEW output directory.

Parent/feature-cache and dataset hashes are checked. Router stage is separate;
resuming does not enable routing or create a calibration certificate. The
original endpoint and accepted parent files are never overwritten.
"""
import argparse,json,torch
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from rgb.runtime import configure_torch,common
from rgb.network import RGBNetwork,RGBConfig
from lightnest_data.core import file_sha
from .cache import read_rows
from .train import Sampler,loss_actor,save_training

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--seed',type=int,required=True,choices=(104729,130363));p.add_argument('--total-steps',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();configure_torch()
    s=torch.load(a.checkpoint,weights_only=True,map_location='cpu');root=a.root.resolve();protocol_hash=file_sha(root/'protocol.json')
    if s['phase']!='actor' or s['protocol_sha256']!=protocol_hash:raise ValueError('wrong phase or protocol')
    if not s['step']<=a.total_steps<=100000:raise ValueError('invalid total step count')
    protocol=json.loads((root/'protocol.json').read_text())
    for k,h in protocol['data'].items():
        if file_sha(root/'data'/k)!=h:raise ValueError('data changed')
    m=RGBNetwork(common.load_parent(a.seed),RGBConfig(**s['rgb_config'])).train();m.load_state_dict(s['state_dict'])
    parent=common.load_parent(a.seed)
    if any(not torch.equal(v,m.red.state_dict()[k]) for k,v in parent.state_dict().items()):raise ValueError('wrong seed/core')
    c=torch.load(root/f'caches/parent-{a.seed}.pt',weights_only=True,map_location='cpu')
    if c['parent_sha256']!=common.sha(common.parent_path(a.seed)):raise ValueError('wrong feature cache')
    for k,h in c['data'].items():
        if file_sha(root/'data'/k)!=h:raise ValueError('cache data changed')
    opt=torch.optim.AdamW(list(m.green.parameters())+list(m.blue.parameters()),lr=3e-4,weight_decay=.01);opt.load_state_dict(s['optimizer'])
    sampler=Sampler(read_rows(root/'data/fit.jsonl'),a.seed);sampler.rng.setstate(s['sampler_rng']);torch.set_rng_state(s['torch_rng']);counts=Counter(s['counts'])
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'steps.jsonl').open('x') as f:
        for step in range(s['step']+1,a.total_steps+1):
            rows=sampler.sample();counts.update(r['id'] for r in rows);opt.zero_grad(set_to_none=True);loss,stats=loss_actor(m,c['entries'],rows)
            loss.backward();torch.nn.utils.clip_grad_norm_(list(m.green.parameters())+list(m.blue.parameters()),1.);opt.step()
            f.write(json.dumps({'step':step,'ids':[r['id'] for r in rows],**stats})+'\n')
    save_training(a.out/'training-state.pt',m,opt,sampler,a.total_steps,'actor',protocol_hash,counts)
    m.eval().save(a.out/'model.pt',note='resumed research actor; routing disabled')
    print('RESUMED',s['step'],a.total_steps,'policy disabled')
if __name__=='__main__':main()
