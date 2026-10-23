"""Write a private candidate and structured trace. Never execute an output."""
import argparse,json,os,hashlib
from pathlib import Path
from . import ROOT
from .model import configure_torch
from .evaluate import build,MODES
from study.evaluate import rollout

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(104729,130363),default=104729)
    ap.add_argument('--strategy',choices=MODES,default='parent64');ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--command',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    b=a.binary.read_bytes()
    if not b or len(b)%4 or len(b)>1024:raise ValueError('input must be aligned RV32, 4..1024 bytes')
    if len(a.command.encode())>96:raise ValueError('command too long')
    configure_torch();pred=build(a.seed,a.strategy)
    rs=rollout(pred,[{'id':'cli','command':a.command,'binary':b.hex()}],batch=1,score=False)
    # No verifier answers entered this prediction. STOP alone is not correctness.
    a.out.mkdir(parents=True,exist_ok=False);r=rs[0]
    for name,body in [('candidate.rv32',bytes.fromhex(r['final'])),('trace.json',json.dumps(r,indent=2).encode())]:
        fd=os.open(a.out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(body)
    print(json.dumps({'candidate':str(a.out/'candidate.rv32'),'strategy':a.strategy,'input_sha256':hashlib.sha256(b).hexdigest(),
         'deployment_approved':False,'host_execution':False,'trace':str(a.out/'trace.json')}))
if __name__=='__main__':main()
