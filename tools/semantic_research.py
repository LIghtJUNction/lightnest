"""Freeze, train and audit external code/description alignment with the RGB core."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightnest.semantics import evaluate, prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--baseline")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--max-bytes", type=int, default=1024)
    parser.add_argument("--from-corpus", help="refreeze unchanged raw pairs under the current implementation")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.out, max_bytes=args.max_bytes, from_corpus=args.from_corpus)
    elif args.phase == "train":
        result = train(args.out, steps=args.steps, seed=args.seed)
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate(args.out, args.checkpoint, baseline=args.baseline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
