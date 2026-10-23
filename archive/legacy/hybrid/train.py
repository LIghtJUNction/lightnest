"""Real supervised actor training and on-policy mode-decision reinforcement.

Only TRAIN records are read. All latent dynamics and action heads are frozen
for policy RL. Rewards use full proposed programs, not one-step confidence.
Cached latent features are exact frozen-state replay, not fake token samples.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time,resource
from copy import deepcopy
import torch
from torch.nn import functional as F
from . import ROOT,BASE
from .model import *
from .prepare import read,write,sha
from rgb.losses import action_correct
from study.evaluate import score_episode

def save(path,obj):
    with path.open('xb') as f:torch.save(obj,f)
    path.chmod(0o600)

def actor_train(seed):
    configure_torch();torch.manual_seed(seed);p=json.loads((ROOT/'protocol.json').read_text());cfg=p['sft']
    m=HybridNetwork(seed).train();initial={k:v.clone() for k,v in m.state_dict().items()}
    rows=read(ROOT/'data/fit.jsonl');rng=random.Random(seed+8711)
    pools={(legacy,edit):[i for i,r in enumerate(rows) if (r['family']=='legacy')==legacy and bool(r['edits'])==edit] for legacy in (False,True) for edit in (False,True)}
    opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    logs=[];cpu=time.process_time();wall=time.perf_counter()
    for step in range(cfg['steps']):
        ids=[rng.choice(pool) for pool in pools.values() for _ in range(2)];es=[rows[i] for i in ids]
        opt.zero_grad(set_to_none=True);outs=m.full(project(es),teacher=True)
        sup=sum(common.supervised_loss(o['scores'],o['bits'],es,m.none_index)[0] for o in outs)/len(outs)
        reg=sum(((o['h']-o['teacher_h'])**2)[o['valid']].mean() for o in outs)/len(outs)
        loss=sup+cfg['latent_mse_weight']*reg
        if not bool(torch.isfinite(loss)):raise FloatingPointError('SFT loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],cfg['clip']);opt.step()
        logs.append({'step':step+1,'indices':ids,'loss':float(loss.detach()),'supervised':float(sup.detach()),'latent_mse':float(reg.detach()),'gradnorm':float(norm)})
        if (step+1)%48==0:print('SFT',seed,step+1,round(logs[-1]['loss'],5),round(time.process_time()-cpu,2),flush=True)
    elapsed=time.process_time()-cpu
    for k,v in m.state_dict().items():
        if k.startswith(('red.','green.')):assert torch.equal(initial[k],v),k
    changed={g:sum(int((initial[k]!=v).sum()) for k,v in m.state_dict().items() if k.startswith(g+'.')) for g in ('red','green','head','link')}
    model={'architecture':ARCH,'seed':seed,'link':m.link.state_dict(),'head':m.head.state_dict(),'optimizer':opt.state_dict(),
       'torch_rng':torch.get_rng_state(),'sampler_rng':rng.getstate(),'steps':cfg['steps'],'protocol_sha256':sha(ROOT/'protocol.json'),
       'baseline_sha256':sha(BASE/f'models/rgb-{seed}.pt')}
    save(ROOT/f'models/actor-{seed}.pt',model)
    write(ROOT/f'results/sft-{seed}.json',{'steps':logs,'cpu_seconds':elapsed,'wall_seconds':time.perf_counter()-wall,'changed':changed,
       'trainable_parameters':sum(q.numel() for q in m.parameters() if q.requires_grad),'parameters':sum(q.numel() for q in m.parameters()),
       'presentations':cfg['steps']*cfg['batch'],'unique_records':len({i for l in logs for i in l['indices']}),
       'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    return m.eval()

class FrozenWorld:
    """TRAIN-only amortized deterministic dynamics; never used for final timing.

Computes all exits once per exact input. This costs more than an early exit:
record both preparation cost and conceptual rollout rounds. Actor fingerprint
is bound for the lifetime of a world. No reward or target is part of cache.
"""
    def __init__(self,m):self.m=m;self.cache={};self.computed_states=0
    @torch.no_grad()
    def get(self,row):
        key=(row['command'],row['binary'])
        if key not in self.cache:
            outs=self.m.full([row]);data=[]
            for o in outs:
                a=action(o['scores'][0],o['bits'][0],self.m.none_index)
                x=o['features'][0].detach().clone()
                data.append((x,a|{'blue':o['depth']<64,'rounds':o['depth'],'text':encode_action(a)}))
            self.cache[key]=data;self.computed_states+=1
        return self.cache[key]
    @torch.no_grad()
    def preload(self,rows,batch=16):
        for start in range(0,len(rows),batch):
            es=project(rows[start:start+batch]);outs=self.m.full(es)
            for j,e in enumerate(es):
                data=[]
                for o in outs:
                    a=action(o['scores'][j],o['bits'][j],self.m.none_index)
                    data.append((o['features'][j].detach().clone(),a|{'blue':o['depth']<64,'rounds':o['depth'],'text':encode_action(a)}))
                self.cache[(e['command'],e['binary'])]=data;self.computed_states+=1

def legal(a,row):
    return (not row['edits']) if a['stop'] else any(e['offset']==a['position'] and e['value']==a['value'] for e in row['edits'])

def policy_sft(seed,world,rows):
    cfg=json.loads((ROOT/'protocol.json').read_text())['mode_sft'];torch.manual_seed(seed+333)
    p=ModePolicy();X=[];Y=[]
    for e in rows:
        for x,a in world.get(project([e])[0])[:-1]:X.append(x);Y.append(int(legal(a,e)))
    x=torch.stack(X);y=torch.tensor(Y);p.normalize_from(x)
    opt=torch.optim.AdamW(p.parameters(),lr=cfg['lr'],weight_decay=.01);gen=torch.Generator().manual_seed(seed+773);losses=[];indices=[];cpu=time.process_time()
    for step in range(cfg['steps']):
        ids=torch.randint(len(y),(cfg['batch'],),generator=gen);indices.append(ids.tolist());opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(p(x[ids]),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    save(ROOT/f'models/policy-sft-{seed}.pt',{'policy':p.state_dict(),'optimizer':opt.state_dict(),'generator':gen.get_state(),'steps':cfg['steps'],
        'actor_sha256':sha(ROOT/f'models/actor-{seed}.pt'),'architecture':ARCH})
    write(ROOT/f'results/mode-sft-{seed}.json',{'losses':losses,'sample_indices':indices,'examples':len(Y),'emit_labels':sum(Y),
        'cpu_seconds':time.process_time()-cpu,'policy_parameters':sum(q.numel() for q in p.parameters())})
    return p.eval()

@torch.no_grad()
def episode(world,policy,e,gen):
    code=bytes.fromhex(e['binary']);st={'code':code,'actions':[],'stop':False,'status':'budget'};sampled=[]
    for step in range(9):
        row={'command':e['command'],'binary':st['code'].hex()}
        try:zs=world.get(row)
        except (ValueError,FloatingPointError,RuntimeError) as exc:st['status']='parse_error';break
        modes=[]
        for k,(x,a) in enumerate(zs):
            if k==len(zs)-1:choose=1
            else:
                logp=policy(x[None]).log_softmax(-1)[0];choose=int(torch.multinomial(logp.exp(),1,generator=gen))
                sampled.append((x.clone(),choose,float(logp[choose])))
            modes.append({'mode':'EMIT' if choose else 'THINK','depth':DEPTHS[k]})
            if choose:break
        a=dict(a,step=step,mode_trace=modes);st['actions'].append(a)
        if a['stop']:st['stop']=True;st['status']='stop';break
        if step==8:break
        pos=a['position'];value=a['value']
        if not 0<=pos<len(st['code']):st['status']='padded_pointer';break
        a['before']=st['code'][pos]
        if value==a['before']:st['status']='no_progress';break
        c=bytearray(st['code']);c[pos]=value;st['code']=bytes(c)
    r=score_episode(e,st)
    reward=(1. if r['strict'] else -1.)-(1. if r['false_edit'] else 0.)
    return r,sampled,reward

def rl_train(seed,world,rows,startpolicy,mode,cost):
    cfg=json.loads((ROOT/'protocol.json').read_text())['rl'];p=deepcopy(startpolicy);ref=deepcopy(startpolicy)
    for q in ref.parameters():q.requires_grad_(False)
    initial={k:v.clone() for k,v in p.state_dict().items()};opt=torch.optim.AdamW(p.parameters(),lr=cfg['lr'],weight_decay=.01)
    rng=random.Random(seed+878);gen=torch.Generator().manual_seed(seed+922);logs=[];rolls=[];cpu=time.process_time();wall=time.perf_counter();maxreplay=0.;vm=0
    for step in range(cfg['updates']):
        xs=[];ys=[];advantages=[];oldlogs=[];rewards=[]
        chosen=rng.sample(range(len(rows)),cfg['tasks_per_update'])
        for idx in chosen:
            group=[]
            for g in range(cfg['rollouts_per_task']):
                r,s,base=episode(world,p,rows[idx],gen);reward=base-cost*r['round_examples']/(9*64);group.append((s,reward))
                vm+=r['execution_calls'];rewards.append(reward)
                rolls.append({'update':step+1,'train_index':idx,'rollout':g,'reward':reward,'record':r,
                    'sampled_modes':[{'choice':a,'old_logp':b,'feature_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()} for x,a,b in s]})
            for j,(s,reward) in enumerate(group):
                adv=reward-sum(v for k,(_,v) in enumerate(group) if k!=j)/(len(group)-1)
                for x,y,old in s:xs.append(x);ys.append(y);advantages.append(adv);oldlogs.append(old)
        if not xs:raise RuntimeError('no sampled mode decisions')
        x=torch.stack(xs);y=torch.tensor(ys);adv=torch.tensor(advantages);opt.zero_grad(set_to_none=True)
        lp=p(x).log_softmax(-1);taken=lp.gather(1,y[:,None]).squeeze(1);err=float((taken.detach()-torch.tensor(oldlogs)).abs().max());maxreplay=max(maxreplay,err)
        # Frozen latent feature replay; only sampled mode log probabilities learn.
        prob=lp.exp();entropy=-(prob*lp).sum(-1).mean()
        with torch.no_grad():ref_lp=ref(x).log_softmax(-1)
        kl=(prob*(lp-ref_lp)).sum(-1).mean()
        # Divide by number of trajectories, not number of variable-length tokens.
        loss=-(taken*adv).sum()/(cfg['tasks_per_update']*cfg['rollouts_per_task'])-cfg['entropy']*entropy+cfg['kl_to_sft']*kl
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step()
        logs.append({'update':step+1,'train_indices':chosen,'mean_reward':sum(rewards)/len(rewards),'loss':float(loss.detach()),'entropy':float(entropy.detach()),'kl':float(kl.detach()),'gradient_norm':float(norm),'sampled_decisions':len(xs),'replay_logp_max_difference':err})
        if (step+1)%24==0:print('RL',seed,mode,step+1,round(logs[-1]['mean_reward'],4),'cache',len(world.cache),flush=True)
    save(ROOT/f'models/policy-{mode}-{seed}.pt',{'policy':p.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':rng.getstate(),
       'generator':gen.get_state(),'steps':cfg['updates'],'actor_sha256':sha(ROOT/f'models/actor-{seed}.pt'),'architecture':ARCH,'cost_weight':cost})
    write(ROOT/f'results/rl-{mode}-{seed}.json',{'steps':logs,'rollouts':rolls,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
        'episodes':len(rolls),'sampled_decisions':sum(z['sampled_decisions'] for z in logs),'vm_calls':vm,'max_replay_logp_error':maxreplay,
        'changed_policy_values':sum(int((initial[k]!=v).sum()) for k,v in p.state_dict().items()),'final_frozen_actor_cache_states':len(world.cache)})

def main(seed):
    configure_torch();m=actor_train(seed)
    for p in m.parameters():p.requires_grad_(False)
    rows=read(ROOT/'data/mode_train.jsonl');world=FrozenWorld(m);cpu=time.process_time();world.preload(rows)
    prep=time.process_time()-cpu;p=policy_sft(seed,world,rows)
    before={k:v.clone() for k,v in m.state_dict().items()}
    # Same immutable deterministic world may be reused; incremental preparation
    # cost differs, so do not compare these learning runtimes as a speed win.
    for mode,cost in [('rl0',0.),('rlcost',.1)]:rl_train(seed,world,rows,p,mode,cost)
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    write(ROOT/f'results/train-complete-{seed}.json',{'actor_unchanged_during_RL':True,'initial_cache_preparation_cpu':prep,'deterministic_full64_states_computed':world.computed_states,
         'cache_bytes_per_state_approx':5*FEATURES*4,'model_files':{p.name:sha(p) for p in (ROOT/'models').glob(f'*-{seed}.pt')}})
    print('COMPLETE',seed,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();main(a.seed)
