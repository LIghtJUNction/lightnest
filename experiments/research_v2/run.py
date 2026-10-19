"""Train fixed endpoints, calibrate on public data, then test frozen endpoints.

Run train for BOTH seeds, then develop for BOTH seeds, then freeze, then test.
No holdout labels are read by training/calibration/selection functions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.nn import functional as F
from . import ARTIFACT_ROOT, BASE
from .methods import (FrozenFeatures, DenseEditHead, DensePlanner, BackwardSlice,
                      StableExit, baseline, evaluate_plans, candidates, validate,
                      patch_action, stop_action, configure_torch)
from .prepare import read, write, sha
from study.evaluate import rollout, summarize, compare


def state_sha(model):
    h=hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def projection(rows):
    return [{'command':r['command'],'binary':r['binary']} for r in rows]


def train(root, seed):
    configure_torch()
    cfg=json.loads((root/'protocol.json').read_text())['training']
    assert sha(ARTIFACT_ROOT/'data/fit.jsonl')==json.loads((root/'protocol.json').read_text())['inputs']['fit']
    rows=read(ARTIFACT_ROOT/'data/fit.jsonl')
    extractor=FrozenFeatures(seed)
    parent_before=state_sha(extractor.red)
    cached={16:[],64:[]}
    cpu=time.process_time()
    wall=time.perf_counter()
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        features=extractor.extract(projection(batch))
        for depth in (16,64):
            for row,(x,ids) in zip(batch,features[depth],strict=True):
                truth={e['offset']//4 for e in row['edits']}
                if not truth.issubset(ids):
                    raise ValueError('Training target outside declared candidate type.')
                y=torch.tensor([int(i in truth) for i in ids],dtype=torch.float32)
                cached[depth].append((x,y))
        if start % 512==0:
            print('FEATURES',seed,start,'/',len(rows),flush=True)
    feature_cost={'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
                  'states':len(rows),'candidate_nodes':sum(len(x) for x,y in cached[16]),
                  'cached_bytes':sum(x.numel()*x.element_size()+y.numel()*y.element_size() for d in cached for x,y in cached[d])}
    for depth in (16,64):
        torch.manual_seed(seed+404)
        head=DenseEditHead()
        xall=torch.cat([x for x,y in cached[depth] if len(x)])
        head.mean.copy_(xall.mean(0))
        head.scale.copy_(xall.std(0).clamp_min(.03))
        del xall
        initial=state_sha(head)
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=.01)
        rng=random.Random(seed+20261020)
        logs=[]
        shown=[]
        cpu=time.process_time()
        wall=time.perf_counter()
        head.train()
        for step in range(cfg['steps']):
            chosen=[rng.randrange(len(rows)) for _ in range(cfg['batch_states'])]
            shown.extend(chosen)
            active=[cached[depth][i] for i in chosen if len(cached[depth][i][0])]
            x=torch.cat([a for a,b in active])
            y=torch.cat([b for a,b in active])
            weights=torch.cat([torch.full((len(b),),1/len(b)) for a,b in active])
            optimizer.zero_grad(set_to_none=True)
            logits=head(x)
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=torch.tensor(2.))*weights).sum()/len(chosen)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
            logs.append({'step':step+1,'sample_indices':chosen,'loss':float(loss.detach()),
                         'gradient_norm':float(norm),'eligible_states':len(active),'candidate_nodes':len(x)})
            if (step+1)%192==0:
                print('TRAIN',seed,depth,step+1,'loss',round(float(loss.detach()),5),flush=True)
        head.eval()
        dest=root/'models'/f'dense{depth}-{seed}.pt'
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'sampler_rng':rng.getstate(),
                    'torch_rng':torch.get_rng_state(),'seed':seed,'depth':depth,'steps':cfg['steps'],
                    'parent_sha256':sha(BASE/f'models/rgb-{seed}.pt')},dest)
        write(root/f'train-{seed}-{depth}.json',{'seed':seed,'depth':depth,'steps':logs,
             'formal_updates':cfg['steps'],'state_presentations':len(shown),'unique_states':len(set(shown)),
             'unique_groups':len({rows[i]['pair_id'] for i in shown}),
             'head_parameters':sum(p.numel() for p in head.parameters()),
             'initial_sha':initial,'final_sha':state_sha(head),
             'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
             'checkpoint_sha256':sha(dest)})
    assert state_sha(extractor.red)==parent_before
    write(root/f'train-complete-{seed}.json',{'features':feature_cost,'frozen_parent_unchanged':True,
          'parent_state_sha256':parent_before,'training_rows':len(rows),
          'sampling_equal_between_depths':True,'formal_updates':2*cfg['steps']})


def load_head(root, seed, depth):
    payload=torch.load(root/'models'/f'dense{depth}-{seed}.pt',weights_only=True,map_location='cpu')
    if payload['parent_sha256']!=sha(BASE/f'models/rgb-{seed}.pt'):
        raise ValueError('Wrong parent for this classifier.')
    head=DenseEditHead()
    head.load_state_dict(payload['head'])
    return head.eval()


def measure(func):
    cpu=time.process_time()
    wall=time.perf_counter()
    result=func()
    return result,{'cpu_seconds_including_score':time.process_time()-cpu,
                  'wall_seconds_including_score':time.perf_counter()-wall}


class CalibrationLogits:
    """Public calibration ONLY. Not used for actual final inference or timing."""
    def __init__(self, rows, features, head):
        self.cache={}
        with torch.no_grad():
            for row,(x,ids) in zip(rows,features,strict=True):
                self.cache[row['command'],row['binary']]=(ids,head(x).tolist())
        self.threshold=0.
    def plan(self,rows):
        out=[]
        for row in rows:
            ids,values=self.cache[row['command'],row['binary']]
            code=bytes.fromhex(row['binary'])
            out.append([patch_action(code,i) for i,v in zip(ids,values,strict=True) if v>self.threshold]+[stop_action()])
        return out


def develop(root,seed):
    configure_torch()
    assert (root/f'train-complete-{seed}.json').exists()
    protocol=json.loads((root/'protocol.json').read_text())
    es=read(BASE/'baseline_data/calibration.jsonl')
    parent=baseline(seed)
    parent_records=rollout(parent,es)
    extractor=FrozenFeatures(seed)
    selections={}
    choices={}
    for threshold in protocol['calibration']['stable_thresholds']:
        records,cost=measure(lambda:rollout(StableExit(extractor,threshold),es))
        comp=compare(parent_records,records)
        su=summarize(records)['all']
        item={'threshold':threshold,'summary':su,'comparison':comp,'cost':cost,
              'qualified':not comp['lost_success_ids'] and not comp['added_false_edit_ids']}
        choices.setdefault('stable_exit',[]).append(item)
        print('CAL',seed,'stable',threshold,su['strict_edits'],su['false_edits'],len(comp['lost_success_ids']),flush=True)
    feasible=[r for r in choices['stable_exit'] if r['qualified']]
    best=min(feasible,key=lambda r:(r['summary']['round_examples'],r['threshold'])) if feasible else None
    selections['stable_exit']={'threshold':best['threshold'] if best else 1.01,
                               'calibration_preserved':best is not None}
    features=extractor.extract(projection(es))
    for depth in (16,64):
        head=load_head(root,seed,depth)
        planner=CalibrationLogits(es,features[depth],head)
        for threshold in protocol['calibration']['dense_thresholds']:
            planner.threshold=threshold
            records=evaluate_plans(planner,es)
            choices.setdefault(f'dense{depth}',[]).append({'threshold':threshold,'summary':summarize(records)['all'],
                                                        'comparison':compare(parent_records,records)})
        best=min(choices[f'dense{depth}'],key=lambda r:(r['summary']['false_edits'],-r['summary']['strict_edits'],
                  -r['summary']['strict_total'],abs(r['threshold'])))
        selections[f'dense{depth}']={'threshold':best['threshold']}
        print('CAL',seed,'dense',depth,best['threshold'],best['summary'],flush=True)
    write(root/f'calibration-{seed}.json',{'source':'previously public calibration, not independent holdout',
          'selections':selections,'choices':choices})
    reg=read(BASE/'baseline_data/regression.jsonl')
    parent_records=None
    for method in protocol['methods']:
        obj=build(root,seed,method,selections,extractor)
        runner=rollout if method in ('C64','stable_exit') else evaluate_plans
        records,cost=measure(lambda:runner(obj,reg,batch=16))
        summary=summarize(records)
        if method=='C64':
            parent_records=records
        comp=compare(parent_records,records)
        write(root/f'development-{seed}-{method}.json',{'seed':seed,'method':method,'split':'public_regression',
              'records':records,'summary':summary,'comparison':comp,'cost':cost})
        print('DEV',seed,method,summary['all']['strict_edits'],'false',summary['all']['false_edits'],
              'lost',len(comp['lost_success_ids']),'cpu',round(cost['cpu_seconds_including_score'],2),flush=True)


def build(root,seed,method,selections,extractor=None):
    if method=='C64':
        return baseline(seed)
    if method=='backward_slice':
        return BackwardSlice()
    extractor=extractor or FrozenFeatures(seed)
    if method=='stable_exit':
        return StableExit(extractor,selections[method]['threshold'])
    depth=int(method.removeprefix('dense'))
    return DensePlanner(extractor,load_head(root,seed,depth),depth,selections[method]['threshold'])


def freeze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    candidates_=[]
    for method in protocol['methods']:
        rows=[json.loads((root/f'development-{seed}-{method}.json').read_text()) for seed in protocol['seeds']]
        feasible=all(not r['comparison']['lost_success_ids'] and not r['comparison']['added_false_edit_ids'] for r in rows)
        score=sum(r['summary']['all']['strict_total'] for r in rows)
        # Runtime including score is a development tie-break only, not reported as an inference speedup.
        duration=sum(r['cost']['wall_seconds_including_score'] for r in rows)
        candidates_.append({'method':method,'feasible_on_public_regression':feasible,
                            'summed_seed_strict':score,'development_wall_seconds_including_score':duration})
    selected=max(candidates_,key=lambda c:(c['feasible_on_public_regression'],c['summed_seed_strict'],
                                           -c['development_wall_seconds_including_score']))['method']
    neural=[c for c in candidates_ if c['method']!='backward_slice']
    neural_best=max(neural,key=lambda c:(c['feasible_on_public_regression'],c['summed_seed_strict'],
                                        -c['development_wall_seconds_including_score']))['method']
    source={str(p.relative_to(Path(__file__).parents[2])):sha(p) for p in Path(__file__).parent.glob('*.py')}
    objects=[*root.glob('models/*.pt'),*root.glob('calibration-*.json'),root/'holdout.jsonl',root/'protocol.json']
    write(root/'final-freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'holdout_model_queries':0,
         'selected_on_public_data':selected,'best_neural_on_public_data':neural_best,'candidates':candidates_,
         'files':{str(p.relative_to(root)):sha(p) for p in objects},'source':source,
         'auto_deployment':False})
    print('FROZEN_SELECTION',selected,'neural',neural_best,flush=True)


def test(root,seed):
    configure_torch()
    freeze_=json.loads((root/'final-freeze.json').read_text())
    for name,digest in freeze_['files'].items():
        assert sha(root/name)==digest,name
    for name,digest in freeze_['source'].items():
        assert sha(Path(__file__).parents[2]/name)==digest,name
    selections=json.loads((root/f'calibration-{seed}.json').read_text())['selections']
    es=read(root/'holdout.jsonl')
    extractor=FrozenFeatures(seed)
    parent=None
    for method in json.loads((root/'protocol.json').read_text())['methods']:
        obj=build(root,seed,method,selections,extractor)
        runner=rollout if method in ('C64','stable_exit') else evaluate_plans
        records,cost=measure(lambda:runner(obj,es,batch=16))
        su=summarize(records)
        if method=='C64':
            parent=records
        comp=compare(parent,records)
        write(root/f'holdout-{seed}-{method}.json',{'seed':seed,'method':method,'split':'new_holdout',
              'records':records,'summary':su,'comparison':comp,'cost':cost,
              'selected_before_holdout':freeze_['selected_on_public_data']})
        print('TEST',seed,method,'edits',su['all']['strict_edits'],'/',su['all']['edits'],
              'false',su['all']['false_edits'],'pairs',su['all']['pairs_both'],
              'lost',len(comp['lost_success_ids']),'cpu',round(cost['cpu_seconds_including_score'],2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','develop','freeze','test'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[104729,130363])
    args=parser.parse_args()
    if args.phase=='freeze':
        freeze(args.root)
    elif args.seed is None:
        parser.error('--seed required')
    else:
        globals()[args.phase](args.root,args.seed)
