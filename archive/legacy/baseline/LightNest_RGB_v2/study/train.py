"""Train actor/observer first, then freeze both and fit the error router.

Fixed endpoints only; no test/validation reads. All arms see identical sampled
record IDs for the same seed. The router loss cannot alter the actor. The
parent core is frozen, with a byte fingerprint checked before/after updates.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time, resource
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import torch
from torch.nn import functional as F
from rgb.network import RGBNetwork,RGBConfig
from rgb.runtime import common,configure_torch
from rgb.losses import action_correct
from lightnest.growth_patch import state_fingerprint
from lightnest_data.core import file_sha
from .cache import read_rows,collate
ARMS={'g4':(4,True),'zero4':(4,False),'g8':(8,True)}

class Sampler:
    def __init__(self,rows,seed):
        self.rng=random.Random(seed);self.buckets={}
        for r in rows:self.buckets.setdefault((r['family'],bool(r['edits'])),[]).append(r)
        self.families=sorted({r['family'] for r in rows if r['family']!='legacy'})
    def sample(self,batch=16):
        if batch!=16:raise ValueError('fixed experiment batch is 16')
        out=[]
        for i in range(batch):
            fam='legacy' if i<4 else self.rng.choice(self.families)
            out.append(self.rng.choice(self.buckets[(fam,i%2==0)]))
        return out


def loss_actor(model,items,rows):
    ctx,h,teacher,teacher_ok=collate(items,rows,model.config.blue_rounds)
    _,(sc,bi),_=model.fast_readout(ctx,h)
    base,_,_=common.supervised_loss(sc,bi,rows,model.none_index)
    kl=F.kl_div(sc[teacher_ok].log_softmax(-1),teacher[teacher_ok],reduction='batchmean') if bool(teacher_ok.any()) else base*0
    return base+.1*kl,dict(supervised=float(base.detach()),teacher_kl=float(kl.detach()),
                          next_correct=int(action_correct(sc,bi,rows,model.none_index).sum()))


def save_training(path,m,opt,sampler,step,phase,protocol_hash,counts):
    tmp=Path(str(path)+'.tmp')
    torch.save({'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':sampler.rng.getstate(),
                'torch_rng':torch.get_rng_state(),'step':step,'phase':phase,'rgb_config':asdict(m.config),
                'red_config':asdict(m.red.config),'protocol_sha256':protocol_hash,'counts':dict(counts)},tmp)
    tmp.replace(path);path.chmod(0o600)


def train_one(root,seed,arm,items,steps=1024):
    configure_torch();torch.manual_seed(seed)
    out=root/'runs'/f'{arm}-{seed}';out.mkdir(parents=True,exist_ok=False)
    r,g=ARMS[arm];m=RGBNetwork(common.load_parent(seed),RGBConfig(blue_rounds=r,green_enabled=g)).train()
    before=state_fingerprint(m.red)
    opt=torch.optim.AdamW(list(m.green.parameters())+list(m.blue.parameters()),lr=3e-4,weight_decay=.01)
    rows=read_rows(root/'data/fit.jsonl');sampler=Sampler(rows,seed);counts=Counter();cpu=time.process_time()
    init={k:v.clone() for k,v in m.state_dict().items()};protocol_hash=file_sha(root/'protocol.json')
    with (out/'actor-steps.jsonl').open('x') as log:
        for step in range(1,steps+1):
            batch=sampler.sample();counts.update(x['id'] for x in batch)
            opt.zero_grad(set_to_none=True);loss,stats=loss_actor(m,items,batch)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(m.green.parameters())+list(m.blue.parameters()),1.)
            if not bool(torch.isfinite(norm)):raise FloatingPointError('training gradient')
            opt.step();log.write(json.dumps({'step':step,'ids':[x['id'] for x in batch],'loss':float(loss.detach()),**stats})+'\n')
            if step%128==0:log.flush();print('ACTOR',arm,seed,step,round(float(loss.detach()),4),flush=True)
            if step in (steps//2,steps):save_training(out/f'actor-{step}.pt',m,opt,sampler,step,'actor',protocol_hash,counts)
    actor_cpu=time.process_time()-cpu
    assert before==state_fingerprint(m.red)
    # Recover the exact next step independently from the full final checkpoint.
    from copy import deepcopy
    rec=deepcopy(m);ropt=torch.optim.AdamW(list(rec.green.parameters())+list(rec.blue.parameters()),lr=3e-4,weight_decay=.01)
    state=torch.load(out/f'actor-{steps}.pt',weights_only=True,map_location='cpu');rec.load_state_dict(state['state_dict']);ropt.load_state_dict(state['optimizer'])
    rs=Sampler(rows,seed);rs.rng.setstate(state['sampler_rng']);test_batch=rs.sample()
    control=deepcopy(m);co=torch.optim.AdamW(list(control.green.parameters())+list(control.blue.parameters()),lr=3e-4,weight_decay=.01);co.load_state_dict(deepcopy(opt.state_dict()))
    for mm,oo in ((rec,ropt),(control,co)):
        oo.zero_grad(set_to_none=True);ll,_=loss_actor(mm,items,test_batch);ll.backward();torch.nn.utils.clip_grad_norm_(list(mm.green.parameters())+list(mm.blue.parameters()),1.);oo.step()
    resume_equal=all(torch.equal(v,rec.state_dict()[k]) for k,v in control.state_dict().items());assert resume_equal
    del rec,ropt,control,co
    # Blue/green fixed. Use disjoint TRAIN construction groups for router fit.
    m.green.requires_grad_(False);m.blue.requires_grad_(False)
    rr=read_rows(root/'data/router_fit.jsonl');features=[];targets=[]
    def hook(mod,args):features.append(args[0].detach().clone())
    handle=m.router.register_forward_pre_hook(hook);cpu_route=time.process_time()
    with torch.no_grad():
        for i in range(0,len(rr),32):
            b=rr[i:i+32];ctx,h,_,_=collate(items,b,r);_,(sc,bi),_=m.fast_readout(ctx,h)
            targets.append((~action_correct(sc,bi,b,m.none_index)).float())
    handle.remove();X=torch.cat(features);Y=torch.cat(targets);rng=random.Random(seed+90000)
    route_opt=torch.optim.AdamW(m.router.parameters(),lr=1e-3,weight_decay=.01)
    with (out/'router-steps.jsonl').open('x') as log:
        for step in range(256):
            ids=[rng.randrange(len(rr)) for _ in range(32)];route_opt.zero_grad(set_to_none=True)
            logits=m.router(X[ids]).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,Y[ids]);loss.backward();route_opt.step()
            log.write(json.dumps({'step':step+1,'ids':[rr[i]['id'] for i in ids],'bce':float(loss.detach())})+'\n')
    assert before==state_fingerprint(m.red)
    m.eval();m.save(out/'model.pt',note='fixed endpoint; router fitted separately; fast path disabled until external calibration')
    torch.save({'optimizer':route_opt.state_dict(),'rng':rng.getstate(),'step':256,'training_targets':Y,'record_ids':[r['id'] for r in rr]},out/'router-state.pt')
    changed={name:sum(int(torch.count_nonzero(v-init[k])) for k,v in m.state_dict().items() if k.startswith(name+'.')) for name in ('red','green','blue','router')}
    (out/'summary.json').write_text(json.dumps({'seed':seed,'arm':arm,'steps':steps,'actor_presentations':16*steps,'actor_unique_records':len(counts),
        'router_steps':256,'router_presentations':8192,'router_fit_rows':len(rr),'router_target_error_rate':float(Y.mean()),
        'actor_cpu_seconds':actor_cpu,'router_cpu_seconds':time.process_time()-cpu_route,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'changed_parameters':changed,'parent_fingerprint_before':before,'parent_fingerprint_after':state_fingerprint(m.red),'resume_next_step_equal':resume_equal,
        'model_sha256':file_sha(out/'model.pt'),'parameter_counts':m.parameter_counts(),'policy_enabled':False},indent=2))
    print('TRAIN_DONE',arm,seed,round(actor_cpu,2),changed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--seed',type=int,required=True);p.add_argument('--arms',nargs='+',default=list(ARMS));a=p.parse_args();root=a.root.resolve()
    cache=torch.load(root/f'caches/parent-{a.seed}.pt',map_location='cpu',weights_only=True)
    assert cache['parent_sha256']==common.sha(common.parent_path(a.seed))
    for k,h in cache['data'].items():assert file_sha(root/'data'/k)==h
    for arm in a.arms:train_one(root,a.seed,arm,cache['entries'])
