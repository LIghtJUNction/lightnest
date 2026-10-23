import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);a=p.parse_args()
for arm in ('g4','zero4','g8'):
    subprocess.run([sys.executable,'-m','study.evaluate','final','--tag',f'{arm}-{a.seed}'],check=True)
