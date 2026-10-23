"""Tool-augmented semantic loop for unseen typed artifacts."""
from __future__ import annotations
import asyncio, json
from pathlib import Path
import torch
from .artifact_types import detect_kind, inspect_artifact
from .any2any import ArtifactCore, ArtifactRequest, KINDS
from .self_agent import python_structure, module_graph, run_tests
from .tools import ToolCall, ToolRegistry

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
   if set(unexpected) or not set(missing) <= {'copy_query.weight'}: raise ValueError(f'Any2Any checkpoint mismatch: {missing} {unexpected}')
   model.eval()
  for path in files:
   try:
    raw=path.read_bytes()
    if not raw: continue
    kind=detect_kind(raw,path.name)
    if model:
     with torch.inference_mode():
      enc,_=model.encode(raw[:256],ArtifactRequest(kind,'route inspection',path.name));_,guidance=model.think(enc,4);pred=int(model.heads(enc,guidance[:,None])['kind_logits'].argmax(-1)[0]);neural.append({'path':str(path.relative_to(root)),'detected':kind,'neural_kind':KINDS[pred] if pred < len(KINDS) else 'custom'})
    if kind=='text' and path.suffix=='.py': calls.append(ToolCall('python_structure',{'path':str(path)},str(path)))
    elif kind=='elf': calls.append(ToolCall('artifact_info',{'payload':raw,'name':path.name},str(path)))
   except OSError: pass
  results=await self.registry.call_many(calls)
  graph=await self.registry.call(ToolCall('module_graph',{'root':str(root)},'graph'))
  tests=await self.registry.call(ToolCall('run_tests',{'root':str(root)},'tests'))
  return {'root':str(root),'observed_files':len(files),'tool_calls':len(results),'python_facts':[r.output for r in results if r.name=='python_structure' and r.error is None],'elf_facts':[r.output for r in results if r.name=='artifact_info' and r.error is None],'module_graph':graph.output,'tests':tests.output,'neural_routing':neural,'routing_conflicts':[r for r in neural if r['detected']!=r['neural_kind']],'errors':[r.error for r in results if r.error]}

 async def ask(self, root, path, question):
  """Answer a generic source question from fresh tool observations."""
  target=Path(root)/path; fact=await self.registry.call(ToolCall('python_structure',{'path':str(target)},'answer'))
  if fact.error: return {'question':question,'answer':None,'error':fact.error}
  return {'question':question,'answer':fact.output,'evidence':'python_structure','path':path}

def inspect_sync(root): return asyncio.run(SemanticLoop().inspect(root))
