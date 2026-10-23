"""Evaluate Any2Any on the LightNest checkout as a held-out artifact set."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import torch
from torch.nn import functional as F

from lightnest_rgb.any2any import ArtifactCore, ArtifactRequest, kind_index
from lightnest_rgb.artifact_types import detect_kind
from lightnest_rgb.any2any_train import code_summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True); model=ArtifactCore(**payload['config']); model.load_state_dict(payload['state_dict']); model.eval()
    files=[p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '.venv' not in p.parts]
    results=[]; overlap=[]
    with torch.inference_mode():
        for path in files:
            raw=path.read_bytes()
            if not raw: continue
            window=raw[:256]; kind=detect_kind(window,path.name); request=ArtifactRequest(kind,'understand LightNest artifact',path.name)
            encoded,_=model.encode(window,request); abstract,guidance=model.think(encoded,depth=8); heads=model.heads(encoded,guidance[:,None])
            target=torch.tensor(list(window)); predicted=heads['red_logits'].argmax(-1)[0]
            kind_pred=int(heads['kind_logits'].argmax(-1)[0]); row={'path':str(path.relative_to(root)),'kind':kind,'kind_correct':kind_pred==kind_index(kind),'byte_accuracy':float((predicted==target).float().mean()),'loss':float(F.cross_entropy(heads['red_logits'].reshape(-1,257),target)),'abstract_width':abstract.shape[-1],'sha256':hashlib.sha256(raw).hexdigest()}
            if path.suffix == '.py':
                expected=torch.tensor([code_summary(raw)]); row['summary_mae']=float((heads['summary'].cpu()-expected).abs().mean())
            results.append(row)
    by={}
    for row in results:
        m=by.setdefault(row['kind'],{'files':0,'kind_accuracy':0.,'byte_accuracy':0.,'loss':0.});m['files']+=1
        for key in ('kind_accuracy','byte_accuracy','loss'):m[key]+=float(row['kind_correct'] if key=='kind_accuracy' else row[key])
        if 'summary_mae' in row: m['summary_mae']=m.get('summary_mae',0.)+row['summary_mae'];m['summary_files']=m.get('summary_files',0)+1
    for m in by.values():
        for key in ('kind_accuracy','byte_accuracy','loss'):m[key]/=m['files']
        if m.get('summary_files'):m['summary_mae']/=m['summary_files']
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'repo_root':str(root),'files':len(results),'by_kind':by,'rows':results,'training_data_overlap':overlap,'claim':'held-out self inspection; format and byte diagnostics, not proof of semantic self-understanding'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k not in ('rows',)},indent=2))

if __name__=='__main__': main()
