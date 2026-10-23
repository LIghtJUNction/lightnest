"""Tool-augmented semantic loop for unseen typed artifacts."""
from __future__ import annotations
import asyncio, json
from pathlib import Path
from .artifact_types import detect_kind, inspect_artifact
from .self_agent import python_structure, module_graph, run_tests
from .tools import ToolCall, ToolRegistry

class SemanticLoop:
 def __init__(self):
  self.registry=ToolRegistry()
  self.registry.register('python_structure',python_structure)
  self.registry.register('module_graph',module_graph)
  self.registry.register('run_tests',run_tests)
  self.registry.register('artifact_info',lambda payload,name: inspect_artifact(payload,name).__dict__)

 async def inspect(self, root):
  root=Path(root); files=[p for p in root.rglob('*') if p.is_file() and not ({'.git','.venv','archive','__pycache__'} & set(p.parts))][:128]
  calls=[]
  for path in files:
   try:
    raw=path.read_bytes(); kind=detect_kind(raw,path.name)
    if kind=='text' and path.suffix=='.py': calls.append(ToolCall('python_structure',{'path':str(path)},str(path)))
    elif kind=='elf': calls.append(ToolCall('artifact_info',{'payload':raw,'name':path.name},str(path)))
   except OSError: pass
  results=await self.registry.call_many(calls)
  graph=await self.registry.call(ToolCall('module_graph',{'root':str(root)},'graph'))
  tests=await self.registry.call(ToolCall('run_tests',{'root':str(root)},'tests'))
  return {'root':str(root),'observed_files':len(files),'tool_calls':len(results),'python_facts':[r.output for r in results if r.name=='python_structure' and r.error is None],'elf_facts':[r.output for r in results if r.name=='artifact_info' and r.error is None],'module_graph':graph.output,'tests':tests.output,'errors':[r.error for r in results if r.error]}

def inspect_sync(root): return asyncio.run(SemanticLoop().inspect(root))
