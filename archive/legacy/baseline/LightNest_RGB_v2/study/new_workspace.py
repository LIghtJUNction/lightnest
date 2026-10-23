"""Create a clean private replay/training directory without modifying evidence."""
import argparse, shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();source=Path(__file__).resolve().parents[1];out=a.out.resolve()
    if out==source or source in out.parents:
        raise ValueError('choose a new directory outside the delivered archive')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('rgb','runtime','lightnest_data','study','data','fixtures','tests','configs'):
        shutil.copytree(source/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('protocol.json','requirements.txt','SOURCE_PROVENANCE.json','SOURCE_PROVENANCE_v0.json'):
        shutil.copy2(source/name,out/name)
    (out/'results').mkdir()
    print('Created',out,'with training source and consumed test data. No automatic training or test execution.')

if __name__=='__main__':main()
