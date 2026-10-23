"""Freeze fresh native tasks, compare raw neural predictions and verified decoding."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from lightnest_native.corpus import FAMILIES,group,jsonl,outside,read,sha,write
from lightnest_native.model import load
from lightnest_native.repair import VerifiedRepair
from lightnest_native.sandbox import Sandbox
from lightnest_native.train import projected,sources


def prepare(out,models,previous):
    out=outside(out);out.mkdir(parents=True,exist_ok=False)
    sandbox=Sandbox(out/'construction-sandbox');records=[];known=set()
    for split in ('train','validation','test'):
        for row in read(previous/'data'/f'{split}.jsonl'):
            for key in ('binary','target'):known.add(hashlib.sha256(bytes.fromhex(row[key])).hexdigest())
    for fi,family in enumerate(FAMILIES):
        got=attempt=0
        while got<16:
            rows=group(family,400000+fi*10000+attempt,'repair-frozen',sandbox);attempt+=1
            if attempt>10000:raise ValueError('construction exhausted')
            if rows is None:continue
            digests={hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in rows for k in ('binary','target')}
            if digests&known:continue
            known.update(digests);records.extend(rows);got+=1
    jsonl(out/'test.jsonl',records)
    protocol={'phase':'post-neural diagnostic, fresh frozen validation-decoder evaluation',
              'test_model_queries':0,'rows':len(records),'data_sha256':sha(out/'test.jsonl'),
              'models':{str(p.resolve()):sha(p) for p in models},'source':sources(),'driver_sha256':sha(Path(__file__)),
              'prior_test_status':'128-task neural test already consumed; these groups are new seeds in the same families',
              'settings':{'candidates':256,'events':16,'wall_seconds':3.0},'thresholds':'unchanged neural 0.5',
              'metrics':'exact bytes + STOP + hidden native behavior; raw neural and verified decoder separately',
              'scope':'finite examples, not a proof; no new neural updates'}
    write(out/'protocol.json',protocol);print('PREPARED',len(records),flush=True)


def evaluate(out):
    out=outside(out);protocol=json.loads((out/'protocol.json').read_text())
    if protocol['source']!=sources() or protocol['data_sha256']!=sha(out/'test.jsonl') or protocol['driver_sha256']!=sha(Path(__file__)):
        raise ValueError('frozen source/data changed')
    torch.set_num_threads(1);rows=read(out/'test.jsonl');summaries={}
    for model_index,(path,digest) in enumerate(protocol['models'].items()):
        if sha(path)!=digest:raise ValueError('frozen model changed')
        model=load(path);sandbox=Sandbox(out/f'evaluation-sandbox-{model_index}')
        repair=VerifiedRepair(model,sandbox,**protocol['settings']);records=[]
        for i,row in enumerate(rows):
            request=projected([row])[0]
            predictions={'neural':model.predict([request])[0],'verified':repair.repair(request)}
            record={'id':row['id'],'family':row['family'],'need_edit':row['binary']!=row['target'],'methods':{}}
            for method,pred in predictions.items():
                result=sandbox.run(bytes.fromhex(pred['final']),[e['inputs'] for e in row['hidden']],timeout=.2)
                behavior=result['status']=='ok' and result['outputs']==[e['output'] for e in row['hidden']]
                record['methods'][method]={'prediction':pred,'hidden_behavior':behavior,'sandbox_status':result['status'],
                    'strict':pred['stop'] and pred['final']==row['target'] and behavior and (record['need_edit'] or not pred['patches']),
                    'false_edit':not record['need_edit'] and bool(pred['patches']),
                    'false_accept':pred['stop'] and not behavior}
            records.append(record)
            if (i+1)%32==0:print('EVAL',Path(path).name,i+1,flush=True)
        summary={}
        for method in ('neural','verified'):
            summary[method]={'rows':len(records),'edits':sum(r['need_edit'] for r in records),
                'strict_edits':sum(r['need_edit'] and r['methods'][method]['strict'] for r in records),
                'strict_total':sum(r['methods'][method]['strict'] for r in records),
                'false_edits':sum(r['methods'][method]['false_edit'] for r in records),
                'false_accepts_on_hidden':sum(r['methods'][method]['false_accept'] for r in records)}
        summary['comparison']={'lost_success_ids':[r['id'] for r in records if r['methods']['neural']['strict'] and not r['methods']['verified']['strict']],
                'added_success_ids':[r['id'] for r in records if not r['methods']['neural']['strict'] and r['methods']['verified']['strict']]}
        summary['verified_candidate_attempts']=sum(r['methods']['verified']['prediction']['candidate_attempts'] for r in records)
        write(out/f'{Path(path).stem}-evaluation.json',{'model_sha256':digest,'summary':summary,'records':records})
        summaries[Path(path).stem]=summary;print('SUMMARY',Path(path).name,json.dumps(summary),flush=True)
    write(out/'summary.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','evaluate'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',type=Path,nargs='+');p.add_argument('--previous',type=Path);a=p.parse_args()
    if a.phase=='prepare':
        if not a.models or not a.previous:p.error('prepare requires --models and --previous')
        prepare(a.out,a.models,a.previous)
    else:evaluate(a.out)
