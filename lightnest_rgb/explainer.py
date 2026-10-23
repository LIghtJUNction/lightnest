"""Generic source-to-text explainer trained on external Python projects."""
from __future__ import annotations

import ast
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

VOCAB = ''.join(chr(i) for i in range(32,127)) + '\n'
BOS, EOS = 127, 128
CHAR = {c:i for i,c in enumerate(VOCAB)}

def summary_text(raw: bytes, name: str) -> str:
    try: tree=ast.parse(raw.decode('utf-8')); nodes=list(ast.walk(tree))
    except (UnicodeDecodeError,SyntaxError): return f"file {name} is not valid python source"
    funcs=[n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))][:8]
    classes=[n.name for n in nodes if isinstance(n,ast.ClassDef)][:8]
    imports=[]
    for n in nodes:
        if isinstance(n,ast.Import): imports.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom) and n.module: imports.append(n.module)
    return f"python source {name}; lines {raw.count(b'\\n')+1}; functions {','.join(funcs) or 'none'}; classes {','.join(classes) or 'none'}; imports {','.join(imports[:8]) or 'none'}"

def encode_text(text): return torch.tensor([BOS,*[CHAR.get(c,CHAR[' ']) for c in text[:240]],EOS],dtype=torch.long)

class CodeExplainer(nn.Module):
    def __init__(self,width=32,max_bytes=65536):
        super().__init__(); self.width=width; self.max_bytes=max_bytes
        self.byte=nn.Embedding(257,width); self.pos=nn.Embedding(256,width)
        self.encoder=nn.GRU(width,width,1,batch_first=True)
        self.decoder=nn.GRU(width,width,2,batch_first=True); self.out=nn.Linear(width,129)
    def encode(self,raw):
        chunks=[]
        starts=list(range(0,len(raw),256)) or [0]
        if len(starts)>2: starts=[starts[0],starts[-1]]
        for start in starts:
            x=torch.tensor(list(raw[start:start+256]),device=self.byte.weight.device)[None]
            p=torch.arange(x.shape[1],device=x.device)[None]
            encoded,_=self.encoder(self.byte(x)+self.pos(p)); chunks.append(encoded.mean(1))
        return torch.stack(chunks,1).mean(1)
    def forward(self,raw,target):
        context=self.encode(raw)[:,None]; tokens=self.byte(target.clamp_max(256))
        hidden=context.transpose(0,1).expand(2,-1,-1).contiguous(); out,_=self.decoder(tokens,hidden)
        return self.out(out)
    @torch.inference_mode()
    def explain(self,raw,max_chars=240):
        context=self.encode(raw)[:,None]; hidden=context.transpose(0,1).expand(2,-1,-1).contiguous(); token=torch.tensor([[BOS]],device=context.device); result=[]
        for _ in range(max_chars):
            out,hidden=self.decoder(self.byte(token[:,-1:]),hidden); value=int(self.out(out[:,-1]).argmax(-1)[0])
            if value==EOS: break
            result.append(VOCAB[value] if value<len(VOCAB) else ' '); token=torch.tensor([[value]],device=context.device)
        return ''.join(result)

def external_python_files(limit=256):
    found=[]
    for root in (Path('/usr/share'),Path('/usr/lib/python3.13/site-packages')):
        if root.exists():
            for path in root.rglob('*.py'):
                try:
                    if path.stat().st_size<=65536: found.append(path)
                except OSError: pass
                if len(found)>=limit: return found
    return found

def train_explainer(out,steps=512,seed=104742,resume=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1);torch.manual_seed(seed)
    samples=[]
    for path in external_python_files():
        try:
            raw=path.read_bytes()
            if raw: samples.append((path.name,raw,summary_text(raw,path.name)))
        except OSError: pass
    if len(samples)<8: raise RuntimeError('not enough external Python sources')
    model=CodeExplainer(width=32);opt=torch.optim.AdamW(model.parameters(),lr=3e-4);rng=random.Random(seed);started=time.perf_counter();losses=[]; start_step=0
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True); model.load_state_dict(state['state_dict']); start_step=state.get('step',0)
    with (out/'updates.jsonl').open('x') as log:
        for step in range(start_step, start_step + steps):
            name,raw,text=rng.choice(samples); target=encode_text(text)[None]
            logits=model(raw,target[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target[:,1:].reshape(-1),ignore_index=0)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            if (step+1)%32==0:
                event={'step':step+1,'loss':sum(losses)/len(losses),'samples':len(samples),'wall_seconds':time.perf_counter()-started};log.write(str(event)+'\n');log.flush();print(event,flush=True);losses=[]
                torch.save({'architecture':'lightnest-explainer-v1','step':step+1,'state_dict':model.state_dict()},out/f'step-{step+1}.pt')
    torch.save({'architecture':'lightnest-explainer-v1','state_dict':model.state_dict(),'metadata':{'seed':seed,'steps':start_step+steps,'samples':len(samples),'status':'research_candidate'}},out/f'explainer-{seed}.pt')
