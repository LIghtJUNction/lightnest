"""Convert an existing LightNest checkpoint to a new complete WGT model directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from lightnest.architecture.any2any import ArtifactCore
from lightnest.architecture.weightlang_io import load_payload, save_model
from lightnest.training.training import load_generator, load_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output exists; choose a new weights/<model> directory')
    if not args.source.is_file(): parser.error('source must be a legacy .pt checkpoint file')
    payload=load_payload(args.source)
    kind=payload['kind']
    if kind=='generator': model,metadata=load_generator(args.source)
    elif kind=='policy': model,metadata=load_policy(args.source)
    elif kind=='artifact-core':
        config=dict(payload['config']); config.setdefault('request_encoding','mean')
        config.setdefault('sequence_limit',payload['state_dict']['position.weight'].shape[0])
        model=ArtifactCore(**config)
        # Conversion must never silently initialize tensors absent from the source.
        model.load_state_dict(payload['state_dict'],strict=True)
        metadata=payload.get('metadata',{})
    else: parser.error('unsupported checkpoint kind: '+kind)
    source=payload['state_dict']; target=model.state_dict()
    if source.keys()!=target.keys():
        parser.error('checkpoint lacks current parameters; strict migration refuses random completion')
    for key in source:
        if source[key].shape!=target[key].shape or source[key].dtype!=target[key].dtype:
            parser.error('tensor contract differs: '+key)
    record={'source':str(args.source.resolve()),'sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'mapping':{key:key for key in source},'transformations':[],
            'claim':'exact parameter transfer; task quality is not revalidated'}
    artifact=save_model(model,args.output,metadata=metadata | {'migration':record},origin='imported')
    for key,value in source.items():
        import torch
        torch.testing.assert_close(value,artifact.module.state_dict()[key],atol=0,rtol=0)
    print(json.dumps({'path':str(artifact.path),'mapped_tensors':len(source),'weights_sha256':artifact.manifest['weights']['sha256']}))


if __name__=='__main__':main()
