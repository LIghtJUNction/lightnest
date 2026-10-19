"""Precompute ONLY frozen-parent states on fit/router_fit records.

These training caches must never be used after updating parent weights. They
are not generic hidden-state caches for similar programs. No calibration/test
labels or states are read here. Cache construction cost is separately reported.
"""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from rgb.runtime import common, configure_torch
from rgb.losses import action_correct
from lightnest_data.core import file_sha


def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]

def build(data,out,seed):
    configure_torch();out.mkdir(parents=True,exist_ok=True)
    path=out/f'parent-{seed}.pt'
    if path.exists():raise FileExistsError(path)
    model=common.load_parent(seed).eval().requires_grad_(False)
    rows=read_rows(data/'fit.jsonl')+read_rows(data/'router_fit.jsonl')
    items={};cpu=time.process_time()
    with torch.no_grad():
        # Shape matching is deterministic and reduces padded inference cost.
        ordered=sorted(rows,key=lambda r:(len(r['binary']),r['id']))
        for start in range(0,len(ordered),32):
            batch=ordered[start:start+32];enc=common.encode(batch,model.config)
            inputs,rel,ctx=model.encode_inputs(*enc);h=inputs;states={}
            for step in range(1,65):
                h=model.shared_step(h,inputs,rel,ctx.valid)
                if step in (4,8):states[step]=h.clone()
            sc,bi=model.decode_state(h,ctx)
            ok=action_correct(sc,bi,batch,model.none_index)
            probs=sc.softmax(-1)
            for j,row in enumerate(batch):
                n=len(row['binary'])//8;l=4*n
                items[row['id']]={
                    'binary':ctx.binary[j,:l].clone(),'byte_embeddings':ctx.byte_embeddings[j,:l].clone(),
                    'slots':ctx.slots[j,:l].clone(),'command':ctx.command[j].clone(),
                    'h4':states[4][j,:n].clone(),'h8':states[8][j,:n].clone(),
                    'teacher_prob':probs[j].clone(),'teacher_ok':bool(ok[j])}
    obj={'entries':items,'seed':seed,'parent_sha256':common.sha(common.parent_path(seed)),
         'data':{k:file_sha(data/k) for k in ('fit.jsonl','router_fit.jsonl')},'cpu_seconds':time.process_time()-cpu}
    torch.save(obj,path)
    (out/f'parent-{seed}.json').write_text(json.dumps({k:v for k,v in obj.items() if k!='entries'}|{'records':len(items),'bytes':path.stat().st_size,'sha256':file_sha(path)},indent=2))
    print('CACHE_DONE',seed,len(items),obj['cpu_seconds'],flush=True)


def collate(items,rows,rounds):
    es=[items[r['id']] for r in rows];L=max(e['binary'].shape[0] for e in es);N=L//4;B=len(es)
    def padded(key,length):
        ex=es[0][key];v=torch.zeros((B,length,*ex.shape[1:]),dtype=ex.dtype)
        for j,e in enumerate(es):v[j,:e[key].shape[0]]=e[key]
        return v
    binary=padded('binary',L);emb=padded('byte_embeddings',L);slots=padded('slots',L);h=padded('h'+str(rounds),N)
    ctx=common.arch.ReadoutContext(binary,emb,slots,torch.stack([e['command'] for e in es]),binary[:,::4].ne(0))
    return ctx,h,torch.stack([e['teacher_prob'] for e in es]),torch.tensor([e['teacher_ok'] for e in es])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,default=Path('caches'));p.add_argument('--seed',type=int,required=True);a=p.parse_args();build(a.data,a.out,a.seed)
