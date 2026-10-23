import argparse
import json
from pathlib import Path
import sys

from .targets import assemble,decode,elf_functions,host_architecture,llvm_targets


def main():
    parser=argparse.ArgumentParser(description='LightNest universal machine-code research; x86-64 training first')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('targets')
    p=sub.add_parser('decode');p.add_argument('--architecture',default=host_architecture());p.add_argument('--hex',required=True)
    p=sub.add_parser('assemble');p.add_argument('--architecture',default=host_architecture())
    p=sub.add_parser('inspect-elf');p.add_argument('path',type=Path)
    p=sub.add_parser('prepare');p.add_argument('--out',type=Path,required=True);p.add_argument('--historical-run',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,choices=(104729,130363),required=True);p.add_argument('--initialization',type=Path,required=True)
    p=sub.add_parser('freeze');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,choices=(104729,130363),required=True)
    p=sub.add_parser('predict');p.add_argument('--checkpoint',type=Path,required=True)
    p=sub.add_parser('repair');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--workdir',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='targets':
        print(json.dumps({'host':host_architecture(),'llvm_targets':llvm_targets(),
                          'note':'registered backend targets; decoding/assembly varies by target and enabled extensions. Training currently targets x86-64 only.'},indent=2))
    elif a.action=='decode':print(json.dumps([i.json() for i in decode(a.architecture,bytes.fromhex(a.hex))],indent=2))
    elif a.action=='assemble':print(json.dumps({'architecture':a.architecture,'binary':assemble(a.architecture,sys.stdin.read()).hex()}))
    elif a.action=='inspect-elf':print(json.dumps(elf_functions(a.path.read_bytes()),indent=2))
    elif a.action=='prepare':
        from .corpus import prepare
        prepare(a.out,a.historical_run)
    elif a.action in ('predict','repair'):
        from .model import load
        model=load(a.checkpoint)
        if a.action=='repair':
            from .corpus import outside
            from .repair import VerifiedRepair
            from .sandbox import Sandbox
            repair=VerifiedRepair(model,Sandbox(outside(a.workdir)))
        for line in sys.stdin:
            row=json.loads(line)
            print(json.dumps(repair.repair(row) if a.action=='repair' else model.predict([row])[0]),flush=True)
    else:
        from .train import train,freeze,evaluate
        if a.action=='train':train(a.out,a.seed,a.initialization)
        elif a.action=='freeze':freeze(a.out)
        else:evaluate(a.out,a.seed)


if __name__=='__main__':main()
