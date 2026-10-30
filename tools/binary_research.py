"""Freeze external bytecode, train RGB semantics/language, and enforce promotion gates."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lightnest.binary_semantics import evaluate, prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--checkpoint')
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--vocab-size', type=int, default=2048)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.phase == 'prepare':
        if not args.parent:
            parser.error('prepare requires --parent frozen source corpus')
        result = prepare(args.out, args.parent, vocab_size=args.vocab_size)
    elif args.phase == 'train':
        result = train(args.out, steps=args.steps, seed=args.seed, batch_size=args.batch_size)
    else:
        if not args.checkpoint:
            parser.error('evaluate requires --checkpoint')
        result = evaluate(args.out, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
