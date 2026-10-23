"""Generic tool-assisted artifact inspection for unseen Python projects."""
from __future__ import annotations
import ast, subprocess
from pathlib import Path
from .tools import ToolCall, ToolRegistry


def python_structure(path):
 raw=Path(path).read_bytes();tree=ast.parse(raw.decode('utf-8'));nodes=list(ast.walk(tree))
 return {'path':str(path),'bytes':len(raw),'lines':raw.count(b'\n')+1,'functions':[n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],'classes':[n.name for n in nodes if isinstance(n,ast.ClassDef)],'imports':[n.name for n in nodes if isinstance(n,ast.Import) for n in n.names]+[n.module for n in nodes if isinstance(n,ast.ImportFrom) and n.module],'calls':sum(isinstance(n,ast.Call) for n in nodes)}

def module_graph(root):
 root=Path(root);graph={}
 paths=[p for p in root.rglob('*.py') if not ({'archive','.venv','.git','__pycache__'} & set(p.parts))][:512]
 for path in paths:
  try: graph[str(path.relative_to(root))]=python_structure(path)['imports']
  except (SyntaxError,UnicodeDecodeError): graph[str(path.relative_to(root))]=['<parse-error>']
 return graph

def run_tests(root):
 result=subprocess.run(['uv','run','--locked','python','-m','unittest','discover','-s','tests'],cwd=root,capture_output=True,text=True,timeout=120)
 return {'returncode':result.returncode,'stdout_tail':result.stdout[-2000:],'stderr_tail':result.stderr[-2000:]}

class SelfAgent:
 def __init__(self):
  self.tools=ToolRegistry();self.tools.register('python_structure',python_structure);self.tools.register('module_graph',module_graph);self.tools.register('run_tests',run_tests)
 async def inspect(self,root):
  root=Path(root);results=await self.tools.call_many([ToolCall('module_graph',{'root':str(root)},'graph'),ToolCall('run_tests',{'root':str(root)},'tests')])
  return {'root':str(root),'tools':self.tools.describe(),'graph':results[0].output,'tests':results[1].output,'errors':[r.error for r in results if r.error]}
