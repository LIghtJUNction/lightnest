"""Pre-holdout amendment: fixed threshold-zero and always-STOP controls.

These are extra inference controls of the SAME frozen trained heads. They add
no training, no data-dependent threshold selection, and no new model seeds.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from . import BASE
from .methods import (FrozenFeatures,DensePlanner,validate,stop_action,evaluate_plans,
                      configure_torch)
from .prepare import read,write,sha
from .run import load_head,build,measure,freeze as original_freeze
from study.evaluate import compare,summarize

BASE_METHODS=('C64','stable_exit','dense16','dense64','backward_slice')
EXTRA_METHODS=('dense16_t0','dense64_t0','always_stop')
ALL_METHODS=BASE_METHODS+EXTRA_METHODS


class AlwaysStop:
    def plan(self,rows):
        for row in rows:
            validate(row)
        return [[stop_action()] for _ in rows]


def build_any(root,seed,method,selections,extractor=None):
    if method=='always_stop':
        return AlwaysStop()
    if method in ('dense16_t0','dense64_t0'):
        depth=16 if method=='dense16_t0' else 64
        return DensePlanner(extractor or FrozenFeatures(seed),load_head(root,seed,depth),depth,0.)
    return build(root,seed,method,selections,extractor)


def evaluate(root,seed,split):
    configure_torch()
    if split=='holdout':
        frozen=json.loads((root/'final-freeze.json').read_text())
        for name,digest in frozen['files'].items():
            assert sha(root/name)==digest
        for name,digest in frozen['source'].items():
            assert sha(Path(__file__).parents[2]/name)==digest
        assert (root/'selection-all-paths.json').exists()
        es=read(root/'holdout.jsonl')
    else:
        assert not (root/'final-freeze.json').exists()
        es=read(BASE/'baseline_data/regression.jsonl')
    reference=json.loads((root/f'{split}-{seed}-C64.json').read_text())['records']
    selections=json.loads((root/f'calibration-{seed}.json').read_text())['selections']
    extractor=FrozenFeatures(seed)
    for method in EXTRA_METHODS:
        model=build_any(root,seed,method,selections,extractor)
        records,cost=measure(lambda:evaluate_plans(model,es,batch=16))
        su=summarize(records)
        comp=compare(reference,records)
        write(root/f'{split}-{seed}-{method}.json',{'seed':seed,'method':method,'split':split,
              'records':records,'summary':su,'comparison':comp,'cost':cost,
              'additional_training_updates':0,'fixed_threshold_control':method.endswith('_t0')})
        print('CONTROL',split,seed,method,'edits',su['all']['strict_edits'],'false',su['all']['false_edits'],
              'lost',len(comp['lost_success_ids']),'cpu',round(cost['cpu_seconds_including_score'],2),flush=True)


def freeze(root):
    assert not (root/'final-freeze.json').exists()
    candidates=[]
    for method in ALL_METHODS:
        objs=[json.loads((root/f'development-{seed}-{method}.json').read_text()) for seed in (104729,130363)]
        candidates.append({'method':method,
           'feasible_on_public_regression':all(not o['comparison']['lost_success_ids'] and
                                               not o['comparison']['added_false_edit_ids'] for o in objs),
           'summed_seed_strict':sum(o['summary']['all']['strict_total'] for o in objs),
           'development_wall_seconds_including_score':sum(o['cost']['wall_seconds_including_score'] for o in objs)})
    rank=lambda c:(c['feasible_on_public_regression'],c['summed_seed_strict'],-c['development_wall_seconds_including_score'])
    selected=max(candidates,key=rank)['method']
    best_neural=max([c for c in candidates if c['method'] not in ('backward_slice','always_stop')],key=rank)['method']
    # Original freeze also seals this module and all added audit/timing sources.
    original_freeze(root)
    write(root/'selection-all-paths.json',{'utc':datetime.now(timezone.utc).isoformat(),'holdout_queries':0,
        'selected_on_public_data':selected,'best_neural_on_public_data':best_neural,'candidates':candidates,
        'amendment_sha256':sha(root/'driver-amendment.json'),'final_freeze_sha256':sha(root/'final-freeze.json'),
        'note':'Fixed-zero and always-STOP controls were registered after public calibration but before any holdout inference.'})
    print('ALL_PATHS_FROZEN',selected,'best neural',best_neural,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','freeze','holdout'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.phase=='freeze':freeze(a.root)
    elif a.seed is None:p.error('--seed required')
    else:evaluate(a.root,a.seed,a.phase)
