"""Make a clean replay workspace. Old holdouts remain consumed, not unseen."""
from pathlib import Path
import argparse,shutil,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 for n in ('rgb','runtime','lightnest_data','study','experiment','baseline_data','data','tests'):
  shutil.copytree(ROOT/n,a.out/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n in ('results','models','policies'):(a.out/n).mkdir()
 shutil.copyfile(ROOT/'protocol.json',a.out/'protocol.json')
 for s in (104729,130363):
  shutil.copyfile(ROOT/f'models/rgb-{s}.pt',a.out/f'models/rgb-{s}.pt')
  shutil.copyfile(ROOT/f'policies/v1-stop-{s}.json',a.out/f'policies/v1-stop-{s}.json')
 (a.out/'REPLAY_NOT_BLIND.json').write_text(json.dumps({'reproduction_only':True,'normal_and_stress_tests_already_consumed':True}))
 print(a.out)
if __name__=='__main__':main()
