"""Tool-augmented semantic loop for unseen typed artifacts."""
from __future__ import annotations
import ast, asyncio, json
from pathlib import Path
import torch
from .artifact_types import detect_kind, inspect_artifact
from .any2any import ArtifactCore, ArtifactRequest, KINDS
from .self_agent import python_structure, module_graph, run_tests
from .tools import ToolCall, ToolRegistry


def _python_summary(raw):
 try:
  tree=ast.parse(raw.decode('utf-8')); nodes=list(ast.walk(tree))
  return [min(len(raw)/65536,1.), min(raw.count(b'\n')/2048,1.),
          min(sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes)/64,1.),
          min(sum(isinstance(n,ast.ClassDef) for n in nodes)/32,1.),
          min(sum(isinstance(n,ast.Import) or isinstance(n,ast.ImportFrom) for n in nodes)/64,1.)]
 except (SyntaxError,UnicodeDecodeError): return None

class SemanticLoop:
 def __init__(self):
  self.registry=ToolRegistry()
  self.registry.register('python_structure',python_structure)
  self.registry.register('module_graph',module_graph)
  self.registry.register('run_tests',run_tests)
  self.registry.register('artifact_info',lambda payload,name: inspect_artifact(payload,name).__dict__)

 async def inspect(self, root, any2any_checkpoint=None):
  root=Path(root); files=[p for p in root.rglob('*') if p.is_file() and not ({'.git','.venv','archive','__pycache__'} & set(p.parts))][:128]
  calls=[]; neural=[]; model=None
  if any2any_checkpoint:
   payload=torch.load(any2any_checkpoint,map_location='cpu',weights_only=True);model=ArtifactCore(**payload['config']);missing,unexpected=model.load_state_dict(payload['state_dict'],strict=False)
   allowed_missing={'copy_query.weight'} | {name for name in missing if name.startswith('budget.')}
   if set(unexpected) or not set(missing) <= allowed_missing: raise ValueError(f'Any2Any checkpoint mismatch: {missing} {unexpected}')
   model.eval()
  for path in files:
   try:
    raw=path.read_bytes()
    if not raw: continue
    kind=detect_kind(raw,path.name)
    if model:
     with torch.inference_mode():
      window=raw[:256]
      enc,_=model.encode(window,ArtifactRequest(kind,'understand and route',path.name))
      abstract,guidance,budget=model.think_budgeted(enc)
      predicted=model.heads(enc,guidance[:,None],budget_logits=budget['logits'])
      pred=int(predicted['kind_logits'].argmax(-1)[0])
      bytes_pred=predicted['red_logits'][0].argmax(-1).tolist()
      record={'path':str(path.relative_to(root)),'detected':kind,
                     'neural_kind':KINDS[pred] if pred < len(KINDS) else 'custom',
                     'byte_accuracy':sum(a == b for a,b in zip(window,bytes_pred,strict=True))/len(window),
                     'summary':[round(float(value),6) for value in predicted['summary'][0]],
                     'budget_index':budget['index'],'budget_depth':budget['depth']}
      target_summary=_python_summary(raw) if path.suffix == '.py' else None
      if target_summary is not None:
       record['summary_target']=target_summary
       record['summary_mae']=sum(abs(a-b) for a,b in zip(record['summary'],target_summary,strict=True))/len(target_summary)
      neural.append(record)
    if kind=='text' and path.suffix=='.py': calls.append(ToolCall('python_structure',{'path':str(path)},str(path)))
    elif kind=='elf': calls.append(ToolCall('artifact_info',{'payload':raw,'name':path.name},str(path)))
   except OSError: pass
  results=await self.registry.call_many(calls)
  graph=await self.registry.call(ToolCall('module_graph',{'root':str(root)},'graph'))
  tests=await self.registry.call(ToolCall('run_tests',{'root':str(root)},'tests'))
  summary_rows=[r for r in neural if 'summary_mae' in r]
  return {'root':str(root),'observed_files':len(files),'tool_calls':len(results),'python_facts':[r.output for r in results if r.name=='python_structure' and r.error is None],'elf_facts':[r.output for r in results if r.name=='artifact_info' and r.error is None],'module_graph':graph.output,'tests':tests.output,'neural_routing':neural,'routing_conflicts':[r for r in neural if r['detected']!=r['neural_kind']],'neural_quality':{'byte_accuracy_mean':sum(r['byte_accuracy'] for r in neural)/len(neural) if neural else None,'python_summary_mae':sum(r['summary_mae'] for r in summary_rows)/len(summary_rows) if summary_rows else None},'errors':[r.error for r in results if r.error]}

 async def ask(self, root, path, question):
  """Answer a generic source question from fresh tool observations."""
  target=Path(root)/path; fact=await self.registry.call(ToolCall('python_structure',{'path':str(target)},'answer'))
  if fact.error: return {'question':question,'answer':None,'error':fact.error}
  return {'question':question,'answer':fact.output,'evidence':'python_structure','path':path}

def inspect_sync(root): return asyncio.run(SemanticLoop().inspect(root))
