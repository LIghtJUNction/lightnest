"""No-attention learned chunking: dense / fixed / learned fast-slow GRU memory.
Uses the existing /usr/bin/python CPU torch. No new dependencies or model downloads.
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

# Three keys, eight values, sixteen irrelevant noise symbols, then control symbols.
KEY, VALUE, END, ASK, ANSWER, BOS, PAD = range(27,34)
VOCAB=34
ARMS=['dense','fixed','learned']
NAMES=[f'k{i}' for i in range(3)]+[f'v{i}' for i in range(8)]+[f'n{i}' for i in range(16)]+['KEY','VALUE','END','ASK','ANSWER','BOS','PAD']


def example(rng,noise=4):
    order=rng.permutation(3).tolist()+rng.integers(0,3,size=int(rng.integers(0,4))).tolist()
    latest={}
    tokens=[BOS]
    for key in order:
        value=int(rng.integers(0,8))+3
        latest[key]=value
        before=(rng.integers(0,16,size=int(rng.integers(noise+1)))+11).tolist()
        after=(rng.integers(0,16,size=int(rng.integers(noise+1)))+11).tolist()
        tokens += [KEY,key]+before+[VALUE,value]+after+[END]
    prefix=tokens.copy()
    questions=rng.permutation(3).tolist()
    answer_positions=[]
    labels=[]
    for key in questions:
        tokens += [ASK,key,ANSWER]
        answer_positions.append(len(tokens))
        labels.append(latest[key])
        tokens.append(latest[key])
    return dict(tokens=tokens,prefix=prefix,questions=questions,labels=labels,answer_positions=answer_positions)


def make_batch(rng,size,noise=4):
    examples=[example(rng,noise) for _ in range(size)]
    length=max(len(e['tokens']) for e in examples)
    x=torch.full((size,length),PAD,dtype=torch.long)
    targets=torch.full((size,length),-100,dtype=torch.long)
    valid=torch.zeros(size,length,dtype=torch.bool)
    for b,e in enumerate(examples):
        x[b,:len(e['tokens'])]=torch.tensor(e['tokens'])
        valid[b,:len(e['tokens'])]=True
        for pos,target in zip(e['answer_positions'],e['labels']):
            targets[b,pos-1]=target
    return x,targets,valid,examples


class Memory(nn.Module):
    def __init__(self,arm,hidden=48,period=8,max_chunk=32):
        super().__init__()
        self.arm,self.hidden_size,self.period,self.max_chunk=arm,hidden,period,max_chunk
        self.embedding=nn.Embedding(VOCAB,24)
        self.fast=nn.GRUCell(24+hidden,hidden)
        self.slow=nn.GRUCell(hidden,hidden)
        self.boundary=nn.Linear(2*hidden+24,1)
        nn.init.normal_(self.boundary.weight,std=0.01)
        nn.init.constant_(self.boundary.bias,math.log((1/period)/(1-1/period)))
        self.output=nn.Linear(2*hidden,VOCAB)
    def initial(self,batch):
        zero=self.embedding.weight.new_zeros(batch,self.hidden_size)
        return zero,zero.clone(),torch.zeros(batch,1)
    def step(self,token,state,sparse=False):
        fast,slow,age=state
        emb=self.embedding(token)
        current=self.fast(torch.cat((emb,slow),dim=-1),fast)
        age=age+1
        if self.arm=='dense':
            boundary=torch.ones_like(age)
        elif self.arm=='fixed':
            boundary=(age>=self.period).float()
        else:
            probability=torch.sigmoid(self.boundary(torch.cat((current,slow,emb),dim=-1)))
            hard=(probability>=0.5).float()
            boundary=hard+(probability-probability.detach()) if self.training else hard
            boundary=torch.where(age>=self.max_chunk,torch.ones_like(boundary),boundary)
        if sparse:
            # Actual conditional slow-cell execution, used in evaluation/streaming.
            active=torch.where(boundary[:,0]>0.5)[0]
            updated=slow.clone()
            if len(active):
                updated[active]=self.slow(current[active],slow[active])
        else:
            # STE training requires computing the candidate even for a hard closed gate.
            candidate=self.slow(current,slow)
            updated=boundary*candidate+(1-boundary)*slow
        # Read before resetting the fast state, so local details remain usable this step.
        logits=self.output(torch.cat((current,updated),dim=-1))
        if self.arm=='dense':
            next_fast=current
        else:
            next_fast=(1-boundary)*current
        age=(1-boundary.detach())*age
        return logits,(next_fast,updated,age),boundary[:,0]
    def forward(self,x,valid):
        state=self.initial(len(x))
        outputs,boundaries=[],[]
        for t in range(x.shape[1]):
            logits,state,b=self.step(x[:,t],state)
            outputs.append(logits);boundaries.append(b)
        boundaries=torch.stack(boundaries,dim=1)
        rate=(boundaries*valid).sum(1)/valid.sum(1)
        return torch.stack(outputs,dim=1),rate,boundaries


@torch.no_grad()
def evaluate(model,examples):
    model.eval()
    token_correct=exact=updates=tokens=0
    role_counts={};role_updates={}
    lengths=[];demos=[]
    # Process each sequence without padding, retaining only fixed-size recurrent state.
    for e in examples:
        state=model.initial(1)
        stream=[];bounds=[];predictions=[];age=0
        def consume(token):
            nonlocal state,tokens,updates,age
            logits,state,b=model.step(torch.tensor([token]),state,sparse=True)
            written=bool(b.item()>0.5)
            stream.append(token);bounds.append(int(written))
            tokens+=1;updates+=int(written);age+=1
            role=NAMES[token] if token>=27 else ('key' if token<3 else 'value' if token<11 else 'noise')
            role_counts[role]=role_counts.get(role,0)+1
            role_updates[role]=role_updates.get(role,0)+int(written)
            if written:lengths.append(age);age=0
            return logits
        for token in e['prefix']:consume(token)
        for key in e['questions']:
            consume(ASK);consume(key)
            logits=consume(ANSWER)
            prediction=int(logits.argmax(-1))
            predictions.append(prediction)
            consume(prediction)  # Never feed a gold answer during rollout.
        correct=[a==b for a,b in zip(predictions,e['labels'])]
        token_correct+=sum(correct);exact+=all(correct)
        if len(demos)<3:
            demos.append({'tokens':[NAMES[t] for t in stream], 'boundaries':bounds,
                          'expected':[NAMES[t] for t in e['labels']], 'predicted':[NAMES[t] for t in predictions]})
    model.train()
    return {'answer_accuracy':token_correct/(3*len(examples)),'exact_match':exact/len(examples),
            'slow_update_fraction':updates/tokens,'tokens_per_slow_update':tokens/max(1,updates),
            'completed_block_length_mean':float(np.mean(lengths)) if lengths else 0,
            'completed_block_length_std':float(np.std(lengths)) if lengths else 0,
            'boundary_rate_by_role':{k:role_updates[k]/v for k,v in role_counts.items()},
            'examples':len(examples),'tokens_read':tokens,'slow_cell_rows_executed':updates,
            'demos':demos}


def verify():
    torch.manual_seed(77)
    x,targets,valid,_=make_batch(np.random.default_rng(9),2)
    for arm in ARMS:
        model=Memory(arm)
        logits,rate,bounds=model(x,valid)
        assert torch.all((bounds.detach()==0)|(bounds.detach()==1))
        changed=x.clone();changed[:,10:]=torch.randint(VOCAB,changed[:,10:].shape)
        logits2,_,_=model(changed,valid)
        torch.testing.assert_close(logits[:,:10],logits2[:,:10])
        model.eval()
        full,_,_=model(x,valid)
        state=model.initial(len(x));sparse=[]
        for t in range(x.shape[1]):
            out,state,_=model.step(x[:,t],state,sparse=True);sparse.append(out)
        torch.testing.assert_close(full,torch.stack(sparse,dim=1),atol=1e-6,rtol=1e-5)
        if arm=='learned':
            model.train()
            logits,_,_=model(x,valid)
            F.cross_entropy(logits.flatten(0,1),targets.flatten()).backward()
            assert model.boundary.weight.grad is not None and model.boundary.weight.grad.norm()>0
        if arm=='fixed':
            expected=(torch.arange(x.shape[1])+1)%8==0
            assert torch.equal(bounds.bool(),expected.expand_as(bounds))
    e=example(np.random.default_rng(4))
    latest={}
    for i,t in enumerate(e['prefix']):
        if t==KEY:key=e['prefix'][i+1]
        if t==VALUE:latest[key]=e['prefix'][i+1]
    assert [latest[k] for k in e['questions']]==e['labels']
    assert not any('attention' in name.lower() or 'qkv' in name.lower() for name,_ in model.named_modules())
    print('Verified: causal stream; binary boundaries; answer-loss gradient to gate; sparse execution equals dense candidates; fixed schedule; record labels; no attention.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--hidden',type=int,default=48)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--eval-size',type=int,default=96)
    p.add_argument('--period',type=int,default=8)
    p.add_argument('--budget-weight',type=float,default=0.02)
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--output',default='results/dynamic-chunks')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    verify()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    config=vars(args)|{'torch':torch.__version__,'device':'cpu','training_noise_max':4,'ood_noise_max':12,
        'target_boundary_rate':1/args.period,'boundary':'causal hard threshold, sigmoid straight-through gradient; no boundary labels; max chunk 32',
        'loss':'query-answer CE + budget_weight * mean((boundary_rate / target_rate - 1)^2)',
        'initialization':'identical parameters in each arm at a given seed; identical online data stream',
        'budget_caveat':'fixed uses nominal period; learned is encouraged, not forced, to achieve same write budget',
        'compute_caveat':'all candidates computed for STE training; only closed chunks execute slow GRU in evaluation; no speedup claim'}
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    tests={name:[example(np.random.default_rng(80000+i*10000+j),noise) for j in range(args.eval_size)]
           for i,(name,noise) in enumerate([('id',4),('long_noise',12)])}
    records=[]
    for index,seed in enumerate(args.seeds):
        order=args.arms[index%len(args.arms):]+args.arms[:index%len(args.arms)]
        for arm in order:
            torch.manual_seed(seed)
            model=Memory(arm,args.hidden,args.period)
            optim=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.001)
            rng=np.random.default_rng(1000+seed)
            checkpoints=[];start=time.perf_counter()
            print(f'START seed={seed} arm={arm} parameters={sum(p.numel() for p in model.parameters())}',flush=True)
            for step in range(1,args.steps+1):
                x,targets,valid,_=make_batch(rng,args.batch)
                optim.zero_grad(set_to_none=True)
                logits,rate,_=model(x,valid)
                ce=F.cross_entropy(logits.flatten(0,1),targets.flatten())
                budget=((rate*args.period-1)**2).mean() if arm=='learned' else ce.new_zeros(())
                loss=ce+args.budget_weight*budget
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                lr=0.003*min(1,step/30)*(0.3+0.7*0.5*(1+math.cos(math.pi*step/args.steps)))
                for group in optim.param_groups:group['lr']=lr
                optim.step()
                if step%args.log_every==0 or step==args.steps:
                    brief={'step':step,'train_answer_ce':float(ce.detach()),'train_write_rate':float(rate.mean().detach())}
                    checkpoints.append(brief)
                    print(json.dumps({'arm':arm,'seed':seed,**brief}),flush=True)
            metrics={name:evaluate(model,examples) for name,examples in tests.items()}
            record={'arm':arm,'seed':seed,'steps':args.steps,'parameters':sum(p.numel() for p in model.parameters()),
                    'seconds':time.perf_counter()-start,'metrics':metrics,'checkpoints':checkpoints}
            records.append(record)
            (root/f'{arm}-seed{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
            torch.save(model.state_dict(),root/f'{arm}-seed{seed}.pt')
            (root/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            brief={name:{k:v for k,v in values.items() if k not in ['demos','boundary_rate_by_role']} for name,values in metrics.items()}
            print(json.dumps({'FINISH':arm,'seed':seed,'metrics':brief}),flush=True)
    print('FINISHED '+str(root),flush=True)


if __name__=='__main__':main()
