"""Emit a complete file using a retained experiment model, without label construction."""
import argparse
from pathlib import Path
import json
import torch
from weightlang.modules import load_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',required=True)
    request=parser.add_mutually_exclusive_group(required=True)
    request.add_argument('--exit',type=int)
    request.add_argument('--stdout-hex')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.round.replace('-', '').isalnum(): parser.error('invalid round ID')
    if args.exit is not None:
        if not 0<=args.exit<=255: parser.error('exit must be between 0 and 255')
        values=[0,args.exit,*([0]*8)]
    else:
        try: payload=bytes.fromhex(args.stdout_hex)
        except ValueError: parser.error('stdout must be hexadecimal')
        if len(payload)!=8: parser.error('stdout must contain exactly eight bytes')
        values=[1,0,*payload]
    root=Path(__file__).resolve().parents[1]
    model=load_module(root/'weights/experiments'/args.round/'model',trusted=True).module.eval()
    torch.set_num_threads(1)
    with torch.inference_mode(): logits=model(torch.tensor([values]))[0]
    if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite output')
    stops=(logits[:,8]>=0).nonzero().flatten()
    if not len(stops): raise RuntimeError('model did not stop; output not written')
    length=int(stops[0]); byte_values=((logits[:length,:8]>=0).long()*(1<<torch.arange(8))).sum(-1)
    raw=bytes(byte_values.tolist())
    # Exclusive creation protects existing files, including symlinks.
    with args.output.open('xb') as file: file.write(raw)
    print(json.dumps(dict(bytes=len(raw),output=str(args.output),stopped=True,behavior='not executed or verified')))


if __name__=='__main__': main()
