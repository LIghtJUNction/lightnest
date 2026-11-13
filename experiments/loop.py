"""Execute declared research rounds, skipping only complete immutable archives."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=1,help='maximum new rounds in this invocation')
    args=parser.parse_args()
    if args.limit<1: parser.error('limit must be positive')
    plan=json.loads((ROOT/'experiments/PLAN.json').read_text())
    completed=0
    env=os.environ.copy()
    local_compiler=ROOT.parent/'WeightLang/WeightLang/target/debug/wgt'
    if local_compiler.is_file(): env.setdefault('WGT_COMPILER',str(local_compiler))
    for item in plan['rounds']:
        archive=ROOT/'experiments/archive'/item['id']
        if (archive/'results.json').exists() and (archive/'REPORT.md').exists(): continue
        if archive.exists(): parser.error(f"incomplete round {item['id']}: inspect it before continuing")
        command=[sys.executable,'-m','experiments.active.elf_generation.run','--round',item['id'],
                 '--hypothesis',item['hypothesis'],*item['args']]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,"-m","experiments.summarize"],cwd=ROOT,check=True)
        completed+=1
        if completed>=args.limit: break
    if not completed: print('All declared rounds are archived. Declare the next hypothesis in experiments/PLAN.json.')


if __name__=='__main__': main()
