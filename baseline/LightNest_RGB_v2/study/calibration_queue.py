from pathlib import Path
import argparse, subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);a=p.parse_args()
for arm in ('g4','zero4','g8'):
    tag=f'{arm}-{a.seed}'
    subprocess.run([sys.executable,'-m','study.evaluate','calibrate','--tag',tag],check=True)
