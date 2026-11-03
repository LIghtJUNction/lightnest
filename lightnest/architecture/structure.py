"""Small CPU structural learner for generic unseen source files."""
from __future__ import annotations
import ast, random, time, json
import re, hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def facts(raw):
 try: tree=ast.parse(raw.decode()); nodes=list(ast.walk(tree))
 except Exception:return [0.]*6
 return [min(len(raw)/65536,1),min(raw.count(b'\n')/2048,1),min(sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes)/64,1),min(sum(isinstance(n,ast.ClassDef) for n in nodes)/32,1),min(sum(isinstance(n,(ast.Import,ast.ImportFrom)) for n in nodes)/64,1),min(sum(isinstance(n,ast.Call) for n in nodes)/128,1)]

def synthetic(seed=1,count=2048):
 r=random.Random(seed);out=[]
 for i in range(count):
  funcs=r.randrange(0,64);classes=r.randrange(0,12);imports=r.randrange(0,16);calls=r.randrange(0,128)
  text=''.join(f'import m{i}_{j}\n' for j in range(imports))+''.join(f'class C{i}_{j}:\n    pass\n' for j in range(classes))+''.join(f'def f{i}_{j}(x):\n    return x+{j}\n' for j in range(funcs))+''.join(f'f{i}_{j}(1)\n' for j in range(min(calls,funcs)))
  out.append(text.encode())
 return out

def external_sources(limit=512):
 out=[]
 for root in (Path('/usr/share'),Path('/usr/lib/python3.13/site-packages')):
  if root.exists():
   for path in root.rglob('*.py'):
    try:
     if path.stat().st_size<=65536: out.append(path.read_bytes())
    except OSError: pass
    if len(out)>=limit:return out
 return out

class StructureNet(nn.Module):
    """Architecture compiled from wgt/structure.wgt."""

def lex_tokens(raw):
 text=raw.decode('utf-8','ignore'); tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]',text)[:4096]
 return [int.from_bytes(hashlib.sha256(t.encode()).digest()[:2],'little')%512 for t in tokens] or [0]

def ast_tokens(raw):
 try: tree=ast.parse(raw.decode('utf-8','ignore')); names=[type(n).__name__ for n in ast.walk(tree)]
 except Exception: names=['Invalid']
 return [int.from_bytes(hashlib.sha256(n.encode()).digest()[:2],'little')%512 for n in names] or [0]

class TokenStructureNet(nn.Module):
    """Architecture compiled from wgt/structure.wgt."""

class ASTStructureNet(TokenStructureNet):
    """Architecture compiled from wgt/structure.wgt."""

def train_ast(out,steps=512,seed=104755):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.set_num_threads(1);samples=synthetic(seed)+external_sources();model=ASTStructureNet();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);rng=random.Random(seed);losses=[]
 with (out/'updates.jsonl').open('x') as log:
  for step in range(steps):
   raw=rng.choice(samples);target=torch.tensor([facts(raw)]);loss=F.mse_loss(model(raw),target);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%64==0:log.write(json.dumps({'step':step+1,'loss':sum(losses)/len(losses)})+'\n');log.flush();losses=[]
 _save_checkpoint({'state_dict':model.state_dict(),'seed':seed,'steps':steps},out/f'ast-{seed}.pt')

def train_ast_synthetic(out,steps=1024,seed=104758):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.set_num_threads(1);samples=synthetic(seed,512);model=ASTStructureNet();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);rng=random.Random(seed);losses=[]
 with (out/'updates.jsonl').open('x') as log:
  for step in range(steps):
   raw=rng.choice(samples);target=torch.tensor([facts(raw)]);loss=F.mse_loss(model(raw),target);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%64==0:log.write(json.dumps({'step':step+1,'loss':sum(losses)/len(losses)})+'\n');log.flush();losses=[]
 _save_checkpoint({'state_dict':model.state_dict(),'seed':seed,'steps':steps},out/f'ast-{seed}.pt')

def train(out,steps=1024,seed=104748,resume=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.set_num_threads(1);samples=synthetic(seed)+external_sources()
 model=StructureNet();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);start=0
 if resume:
  state=torch.load(resume,map_location='cpu',weights_only=True);model.load_state_dict(state['state_dict']);start=state.get('step',0)
 rng=random.Random(seed);losses=[]
 with (out/'updates.jsonl').open('x') as log:
  for step in range(start,start+steps):
   raw=rng.choice(samples);target=torch.tensor([facts(raw)]);loss=F.mse_loss(model(raw[:8192]),target);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%64==0:
    log.write(json.dumps({'step':step+1,'loss':sum(losses)/len(losses)})+'\n');log.flush();losses=[]
    _save_checkpoint({'state_dict':model.state_dict(),'step':step+1},out/f'step-{step+1}.pt')
 _save_checkpoint({'state_dict':model.state_dict(),'seed':seed,'steps':start+steps},out/f'structure-{seed}.pt')

def train_tokens(out,steps=1024,seed=104750,resume=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);torch.set_num_threads(1);samples=synthetic(seed)+external_sources();model=TokenStructureNet();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);start=0
 if resume:
  state=torch.load(resume,map_location='cpu',weights_only=True);model.load_state_dict(state['state_dict']);start=state.get('step',0)
 rng=random.Random(seed);losses=[]
 with (out/'updates.jsonl').open('x') as log:
  for step in range(start,start+steps):
   raw=rng.choice(samples);target=torch.tensor([facts(raw)]);loss=F.mse_loss(model(raw),target);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%64==0:
    log.write(json.dumps({'step':step+1,'loss':sum(losses)/len(losses)})+'\n');log.flush();losses=[];_save_checkpoint({'state_dict':model.state_dict(),'step':step+1},out/f'step-{step+1}.pt')
 _save_checkpoint({'state_dict':model.state_dict(),'seed':seed,'steps':start+steps},out/f'tokens-{seed}.pt')


# WGT is the architecture source; host code above handles preprocessing and scheduling.
from pathlib import Path as _WgtPath
from weightlang.tensor_program import bind_tensor_program as _bind_wgt
globals().update(_bind_wgt(_WgtPath(__file__).with_name("wgt") / "structure.wgt", globals()))


def _save_checkpoint(payload, path):
    from lightnest.architecture.weightlang_io import save_model
    torch.save(payload, path)
    name = Path(path).name
    model_class = ASTStructureNet if name.startswith("ast-") else TokenStructureNet if name.startswith("tokens-") else StructureNet
    width = payload["state_dict"].get("token.weight", payload["state_dict"].get("byte.weight")).shape[1]
    # Intermediate token checkpoints use the same explicit tensor namespace.
    if "gru.weight_ih_l1" in payload["state_dict"]:
        model_class = TokenStructureNet
    with torch.random.fork_rng(devices=[]):
        model = model_class(width=width)
        model.load_state_dict(payload["state_dict"], strict=True)
    steps = payload.get("step", payload.get("steps"))
    save_model(model, Path(path).with_suffix(""), kwargs={"width": width},
               metadata={k:v for k,v in payload.items() if k != "state_dict"},
               training={"method": "AdamW", "steps": steps})
