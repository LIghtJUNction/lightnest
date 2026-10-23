"""uv run python -m lightnest_next --help"""
import argparse
import json
from pathlib import Path
import sys

from .data import prepare, output_directory
from .model import load_model


def main():
    parser = argparse.ArgumentParser(description="LightNest RPN3 research; no native candidate execution")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="read original archives and freeze data splits")
    for name in ("hybrid", "dataset", "parents", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p = sub.add_parser("train", help="train a fixed endpoint")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(104729,130363),required=True)
    p = sub.add_parser("freeze", help="freeze both completed models before final evaluation")
    p.add_argument("--out",type=Path,required=True)
    p = sub.add_parser("evaluate", help="evaluate against stable and expanded parents")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(104729,130363),required=True)
    p = sub.add_parser("predict", help="read request objects as JSONL on stdin; return unverified candidates")
    p.add_argument("--checkpoint",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.hybrid,args.dataset,args.parents,args.out)
    elif args.command == "predict":
        model = load_model(args.checkpoint)
        for line in sys.stdin:
            for pred in model.predict([json.loads(line)]):
                print(json.dumps(pred),flush=True)
    else:
        from .training import train,freeze,evaluate
        out = output_directory(args.out)
        if args.command == "train":
            train(out,args.seed)
        elif args.command == "freeze":
            freeze(out)
        else:
            evaluate(out,args.seed)


if __name__ == "__main__":
    main()
