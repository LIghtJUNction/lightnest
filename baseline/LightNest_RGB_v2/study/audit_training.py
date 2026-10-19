"""Read-only audit of actual update streams, frozen core and role separation."""
from collections import Counter
from pathlib import Path
import json, torch
from rgb.runtime import configure_torch,common
from rgb.network import load_rgb
from lightnest.growth_patch import state_fingerprint
from lightnest_data.core import file_sha,neutral_fingerprint
from .cache import read_rows
from .train import Sampler,ARMS

def main(root=Path('.')):
    configure_torch();p=json.loads((root/'protocol.json').read_text());errors=[]
    for rel,h in p['source_sha256'].items():
        if file_sha(root/rel)!=h:errors.append('changed source '+rel)
    for rel,h in p['data'].items():
        if file_sha(root/'data'/rel)!=h:errors.append('changed data '+rel)
    fit=read_rows(root/'data/fit.jsonl');router=read_rows(root/'data/router_fit.jsonl')
    assert not {x['pair_id'] for x in fit}&{x['pair_id'] for x in router}
    check=[]
    for seed in p['seeds']:
        parent=common.load_parent(seed);hf=state_fingerprint(parent);streams=[]
        for arm in ARMS:
            out=root/'runs'/f'{arm}-{seed}';sm=json.loads((out/'summary.json').read_text());sampler=Sampler(fit,seed);cnt=Counter();ids_stream=[]
            lines=(out/'actor-steps.jsonl').read_text().splitlines();assert len(lines)==1024
            for step,line in enumerate(lines,1):
                actual=json.loads(line);ids=[x['id'] for x in sampler.sample()];assert actual['step']==step and actual['ids']==ids
                cnt.update(ids);ids_stream+=ids
            streams.append(ids_stream);model=load_rgb(out/'model.pt');assert state_fingerprint(model.red)==hf
            state=torch.load(out/'actor-1024.pt',weights_only=True,map_location='cpu')
            assert state['step']==1024 and state['counts']==dict(cnt) and state['sampler_rng']==sampler.rng.getstate()
            assert all(int(v['step'])==1024 for v in state['optimizer']['state'].values())
            rid={r['id'] for r in router};rr=[json.loads(x) for x in (out/'router-steps.jsonl').read_text().splitlines()];assert len(rr)==256
            assert all(set(x['ids'])<=rid for x in rr)
            assert sm['model_sha256']==file_sha(out/'model.pt') and sm['resume_next_step_equal']
            check.append({'tag':f'{arm}-{seed}','actor_updates':1024,'actor_presentations':len(ids_stream),'actor_unique_ids':len(cnt),
                          'router_updates':len(rr),'router_presentations':sum(len(x['ids']) for x in rr),'frozen_core_equal':True,
                          'actor_sample_order_reproduced':True,'resume_next_step_equal':True})
        assert streams[0]==streams[1]==streams[2]
    report={'passed':not errors,'errors':errors,'runs':check,'training_source_immutable':not errors,
            'actor_updates':sum(r['actor_updates'] for r in check),'router_updates':sum(r['router_updates'] for r in check),
            'actor_presentations':sum(r['actor_presentations'] for r in check),'router_presentations':sum(r['router_presentations'] for r in check),
            'scope':'same-session read-only code audit, not external institution'}
    (root/'results/training-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
