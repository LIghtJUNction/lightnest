"""Explicit research entry point; default red. Never execute or overwrite output."""
import argparse,json,os
from pathlib import Path
from experiment.run_study import ROOT,load_rgb,run_mode,configure_torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[104729,130363],default=104729)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--request',required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--mode',choices=['red','v1_stop','risk_full','risk_stats','verified'],default='red');a=p.parse_args()
 if a.input.stat().st_size>1024:raise ValueError('maximum1024 bytes')
 raw=a.input.read_bytes();configure_torch();m=load_rgb(ROOT/f'models/rgb-{a.seed}.pt')
 row={'id':'local','command':a.request,'binary':raw.hex()}
 result=run_mode(m,[row],a.seed,a.mode,batch=1,score=False)
 a.out.mkdir(parents=True,exist_ok=False)
 for name,content in [('candidate.bin',bytes.fromhex(result['records'][0]['final'])),
                      ('trace.json',json.dumps(result|{'deployment_approved':False,'not_executed':True},indent=2).encode())]:
  fd=os.open(a.out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'wb') as f:f.write(content)
 print(json.dumps({'path':str(a.out),'mode':a.mode,'not_executed':True,'deployment_approved':False}))
if __name__=='__main__':main()
