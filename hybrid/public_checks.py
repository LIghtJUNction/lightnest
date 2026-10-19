"""Post-primary public ablations and fixed workload timing, not new blind tests."""
import json,time,statistics,hashlib
from copy import deepcopy
import torch
from . import ROOT,BASE
from .model import *
from .prepare import read,write,sha
from .evaluate import build,MODES
from study.evaluate import rollout,summarize,compare

def main():
    configure_torch()
    for s in (104729,130363):
        assert (ROOT/f'results/evaluation-complete-{s}.json').exists(),'finish other workers before timing'
    allrows=read(BASE/'baseline_data/calibration.jsonl');families=['simple','mixed','overwrite','spill','branch','loop','branch_spill','loop_spill']
    workloads={'simple': [r for r in allrows if r['family']=='simple'][:32],
       'balanced':sum(([r for r in allrows if r['family']==fam][:4] for fam in families),[])}
    write(ROOT/'results/public-check-plan.json',{'type':'public diagnostic and timing; no checkpoint or threshold changes',
       'row_ids':{k:[r['id'] for r in v] for k,v in workloads.items()},'repeats':3,'batch':1,
       'timing_excludes':['model load','compile','warmup','VM scoring'],'result_cache':False})
    diags={};timings={};reinferences=[]
    for seed in (104729,130363):
        # Feature-off intervention uses the same trained action head, no retraining.
        m=load_actor(seed);rows=workloads['balanced']
        on=rollout(lambda rs:m.predict(rs,fixed64=True),rows)
        off=rollout(lambda rs:m.predict(rs,fixed64=True,linked=False),rows)
        diags[str(seed)]={'link_on':summarize(on),'link_off':summarize(off),'comparison':compare(off,on),'on_records':on,'off_records':off}
        # Test-time features replay should match a fresh runtime. Public inputs
        # only; stored cache is deliberately not used in timing or final test.
        from .train import FrozenWorld
        w=FrozenWorld(m);w.preload(rows)
        for mode in ('sft','rl0','rlcost'):
            p=load_policy(seed,mode);live=m.predict(project(rows),p)
            for e,a in zip(rows,live,strict=True):
                levels=w.get(project([e])[0]);expected=None
                for k,(x,b) in enumerate(levels):
                    if k==4 or int(p(x[None]).argmax(-1))==1:expected=b;break
                assert all(a[key]==expected[key] for key in ('position','value','stop','rounds'))
            reinferences.append({'seed':seed,'strategy':mode,'public_action_cache_vs_live':len(rows),'matched':True})
        for mode in MODES:
            pred=build(seed,mode)
            for label,es in workloads.items():
                # Warm the immutable parser cache and exact runtime path.
                rollout(pred,es,batch=1,score=False)
        # Rotate full methods, record every repetition, no best-of reporting.
        preds={mode:build(seed,mode) for mode in MODES}
        vals={label:{mode:[] for mode in MODES} for label in workloads}
        for rep in range(3):
            order=MODES[rep:]+MODES[:rep]
            for mode in order:
                for label,es in workloads.items():
                    cpu=time.process_time();wall=time.perf_counter();rs=rollout(preds[mode],es,batch=1,score=False)
                    vals[label][mode].append({'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
                       'actions':sum(len(r['actions']) for r in rs),'rounds':sum(a['rounds'] for r in rs for a in r['actions'])})
        timings[str(seed)]={}
        for label,data in vals.items():
            med={mode:statistics.median(x['wall_seconds'] for x in runs) for mode,runs in data.items()}
            timings[str(seed)][label]={'raw':data,'median_wall':med,'speed_vs_parent':{m:med['parent64']/v for m,v in med.items()},
                     'speed_vs_linked_fixed64':{m:med['sft64']/v for m,v in med.items()},
                     'warning':'different policies may stop early incorrectly; not identical-function acceleration'}
    write(ROOT/'results/public-diagnostics.json',diags)
    write(ROOT/'results/timing.json',timings)
    write(ROOT/'results/reinference.json',{'passed':True,'checks':reinferences})
    print('PUBLIC_CHECKS_COMPLETE',flush=True)
if __name__=='__main__':main()
