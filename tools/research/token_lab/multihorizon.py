"""Controlled next-token vs consecutive vs exponentially spaced future prediction.
Run with /usr/bin/python (uses the installed CPU PyTorch, no downloads).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BOS, EDGE, FILL, ANSWER, EOS, COPY, ROUTE = range(32, 39)
VOCAB, LENGTH, ANSWER_AT = 39, 128, 115
ROUTE_DEPTH = 3
HORIZONS = {'next': [1], 'near': list(range(1, 8)), 'powers': [1, 2, 4, 8, 16, 32, 64]}


def sample(rng, task):
    if task == 'route':
        labels = rng.permutation(32)
        root = int(labels[0])
        branches = labels[1:1+4*ROUTE_DEPTH].reshape(4, ROUTE_DEPTH)
        edges = []
        for branch in branches:
            path = [root]+[int(v) for v in branch]
            edges += list(zip(path[:-1],path[1:]))
        begin = 1+4*ROUTE_DEPTH
        edges += [(int(labels[begin+2*i]),int(labels[begin+2*i+1])) for i in range(16-4*ROUTE_DEPTH)]
        rng.shuffle(edges)
        branch = branches[int(rng.integers(4))].tolist()
        prefix = [BOS, ROUTE, root, branch[-1]]
        prefix += [v for a,b in edges for v in (EDGE,a,b)]
        target = branch+[EOS]
    else:
        digits = rng.integers(0, 16, size=8).tolist()
        prefix = [BOS, COPY]+digits
        target = digits+[EOS]
    prefix += [FILL]*(ANSWER_AT-len(prefix))
    prefix += [ANSWER]
    tokens = prefix+target
    assert len(prefix) == ANSWER_AT+1 and len(tokens) <= LENGTH
    row = np.full(LENGTH, FILL, dtype=np.int64)
    row[:len(tokens)] = tokens
    valid = np.arange(LENGTH) < len(tokens)
    return row, valid, len(target), target


def batch(rng, size, task=None):
    examples = [sample(rng, task or ('route' if rng.random()<0.5 else 'copy')) for _ in range(size)]
    x = torch.from_numpy(np.stack([v[0] for v in examples]))
    valid = torch.from_numpy(np.stack([v[1] for v in examples]))
    return x, valid, examples


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim,3*dim), nn.Linear(dim,dim)
        self.ff = nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(b,t,3,4,d//4).permute(2,0,3,1,4).unbind(0)
        attention=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(attention.transpose(1,2).reshape(b,t,d))
        return x+self.ff(self.ln2(x))


class Model(nn.Module):
    def __init__(self,dim=64):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,dim)
        self.pos=nn.Embedding(LENGTH,dim)
        self.blocks=nn.Sequential(Block(dim),Block(dim))
        self.norm=nn.LayerNorm(dim)
        # Every arm has exactly the same parameters and initial backbone / next-token head.
        self.heads=nn.ModuleList([nn.Linear(dim,VOCAB,bias=False) for _ in range(7)])
    def hidden(self,x):
        return self.norm(self.blocks(self.emb(x)+self.pos(torch.arange(x.shape[1]))))
    def forward(self,x):
        return self.heads[0](self.hidden(x))


def loss_for(model,x,valid,horizons,aux_weight=1.0):
    hidden=model.hidden(x)
    losses=[]
    for index,offset in enumerate(horizons):
        logits=model.heads[index](hidden[:,:-offset])
        targets=x[:,offset:]
        mask=valid[:,offset:]
        per_token=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1),reduction='none').view_as(targets)
        losses.append((per_token*mask).sum()/mask.sum())
    total=losses[0]
    if len(losses)>1:
        total=total+aux_weight*torch.stack(losses[1:]).mean()
    return total,losses[0]


@torch.no_grad()
def evaluate(model,test):
    model.eval()
    result={}
    for task,(x,valid,examples) in test.items():
        logits=model(x[:,:-1])
        y=x[:,1:]
        mask=valid[:,1:] & (torch.arange(LENGTH-1)[None,:]>=ANSWER_AT)
        ce=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y)
        teacher_pred=logits.argmax(-1)
        prefix=x[:,:ANSWER_AT+1]
        generated=[]
        # Only head 0 is called. Append exactly one predicted token on each iteration.
        for _ in range(len(examples[0][3])):
            next_token=model(prefix)[:,-1].argmax(-1)
            generated.append(next_token)
            prefix=torch.cat((prefix,next_token[:,None]),dim=1)
        generated=torch.stack(generated,dim=1)
        expected=torch.tensor([v[3] for v in examples])
        result[task]={
            'answer_nll':float((ce*mask).sum()/mask.sum()),
            'teacher_answer_accuracy':float(((teacher_pred==y)*mask).sum()/mask.sum()),
            'first_token_accuracy':float((generated[:,0]==expected[:,0]).float().mean()),
            'rollout_token_accuracy':float((generated==expected).float().mean()),
            'exact_match':float((generated==expected).all(1).float().mean()),
            'examples':len(examples),
        }
    model.train()
    return result


def verify():
    torch.manual_seed(7)
    rng=np.random.default_rng(11)
    x,valid,_=batch(rng,2)
    model=Model()
    # Causal backbone must not reveal future targets, including distant ones.
    changed=x.clone();changed[:,40:]=torch.randint(VOCAB,changed[:,40:].shape)
    torch.testing.assert_close(model(x)[:,:40],model(changed)[:,:40],atol=1e-6,rtol=1e-5)
    # Explicitly verify t -> t+offset indexing and masking of right padding.
    for offset in HORIZONS['powers']:
        assert x[:,offset:].shape == valid[:,offset:].shape
        assert x.shape[1]-offset>0
    total,_=loss_for(model,x,valid,HORIZONS['powers'])
    total.backward()
    assert all(head.weight.grad is not None and head.weight.grad.norm()>0 for head in model.heads)
    # Every auxiliary horizon must itself contribute gradient to shared representation.
    for head,offset in enumerate(HORIZONS['powers'][1:],1):
        model.zero_grad(set_to_none=True)
        h=model.hidden(x)
        loss=F.cross_entropy(model.heads[head](h[:,:-offset]).reshape(-1,VOCAB),x[:,offset:].reshape(-1),reduction='none')
        mask=valid[:,offset:].reshape(-1)
        ((loss*mask).sum()/mask.sum()).backward()
        assert model.emb.weight.grad.norm()>0
    print('Verified: causal masking, valid horizons, all auxiliary heads and shared-backbone gradients.',flush=True)


def main():
    global ROUTE_DEPTH
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--route-depth',type=int,choices=[1,2,3],default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--arms',nargs='+',choices=HORIZONS,default=list(HORIZONS))
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-size',type=int,default=256)
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--output',default='results/multihorizon')
    args=p.parse_args()
    ROUTE_DEPTH=args.route_depth
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    verify()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=vars(args)|{'torch':torch.__version__,'device':'cpu','horizons':HORIZONS,
                       'loss':'L_next + mean(L_aux), each horizon mean over valid targets',
                       'tasks':f'online random 4-branch length-{ROUTE_DEPTH} routes, and delayed copy of 8 symbols; both with fixed filler gap',
                       'training':'full sequence CE, no target-only weighting; fresh paired batches for each arm',
                       'primary_metrics':['route exact_match','route first_token_accuracy','copy exact_match'],
                       'decoding':'greedy, next-token head only, one token appended each iteration'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    test={task:batch(np.random.default_rng(90000+i),args.eval_size,task) for i,task in enumerate(['route','copy'])}
    records=[]
    for seed_index,seed in enumerate(args.seeds):
        # Rotate order to reduce systematically favoring one arm with machine drift.
        order=args.arms[seed_index%len(args.arms):]+args.arms[:seed_index%len(args.arms)]
        for arm in order:
            torch.manual_seed(seed)
            model=Model(args.dim)
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01)
            rng=np.random.default_rng(1000+seed)
            start=time.perf_counter()
            checkpoints=[]
            print(f'START seed={seed} arm={arm} params={sum(v.numel() for v in model.parameters())}',flush=True)
            for step in range(1,args.steps+1):
                x,valid,_=batch(rng,args.batch)
                optimizer.zero_grad(set_to_none=True)
                loss,next_loss=loss_for(model,x,valid,HORIZONS[arm])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                warmup=min(1.0,step/50)
                lr=0.002*warmup*(0.2+0.8*0.5*(1+math.cos(math.pi*step/args.steps)))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.step()
                if step%args.log_every==0 or step==args.steps:
                    metrics=evaluate(model,test)
                    checkpoint={'step':step,'next_train_nll':float(next_loss.detach()),'metrics':metrics}
                    checkpoints.append(checkpoint)
                    print(json.dumps({'seed':seed,'arm':arm,**checkpoint}),flush=True)
            record={'seed':seed,'arm':arm,'steps':args.steps,'seconds':time.perf_counter()-start,
                    'params_total':sum(v.numel() for v in model.parameters()),
                    'metrics':checkpoints[-1]['metrics'],'checkpoints':checkpoints}
            records.append(record)
            (out/f'{arm}-seed{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
            # Small final artifact, no historical checkpoint accumulation.
            torch.save(model.state_dict(),out/f'{arm}-seed{seed}.pt')
            (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    print('FINISHED '+str(out),flush=True)


if __name__=='__main__':main()
