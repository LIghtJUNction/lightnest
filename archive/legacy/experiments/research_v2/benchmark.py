"""Serial full-request latency; no scores, model loading or result cache.

Warm parse cache and cold parse cache are measured separately. All policies
run on every selected public request, including failures. Policy quality may
be different, so these speed ratios are NOT quality-preserving guarantees.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from . import BASE
from .prepare import read,write
from .methods import common,configure_torch,evaluate_plans
from .controls import build_any, ALL_METHODS
from study.evaluate import rollout


def main(root):
    configure_torch()
    protocol=json.loads((root/'protocol.json').read_text())
    methods=list(ALL_METHODS)
    es=read(BASE/'baseline_data/calibration.jsonl')
    families=sorted({r['family'] for r in es})
    balanced=[r for family in families for r in [e for e in es if e['family']==family][:8]]
    assert len(balanced)==64
    # Explicitly choose two declared warm/cold conditions before any timing.
    measurements=[]
    for seed in protocol['seeds']:
        selections=json.loads((root/f'calibration-{seed}.json').read_text())['selections']
        models={m:build_any(root,seed,m,selections) for m in methods}
        for method,obj in models.items():
            runner=rollout if method in ('C64','stable_exit') else evaluate_plans
            runner(obj,balanced[:8],batch=1,score=False)
        for batch in (1,16):
            for cache in ('warm','cold'):
                entries={m:[] for m in models}
                signatures={}
                for repeat in range(5):
                    order=methods[repeat:]+methods[:repeat]
                    for method in order:
                        obj=models[method]
                        runner=rollout if method in ('C64','stable_exit') else evaluate_plans
                        common.graph.cache_clear()
                        if cache=='warm':
                            # Warm entire policy input path, not a neural result cache.
                            runner(obj,balanced,batch=batch,score=False)
                        cpu=time.process_time();wall=time.perf_counter()
                        out=runner(obj,balanced,batch=batch,score=False)
                        duration=time.perf_counter()-wall
                        entries[method].append({'wall_seconds':duration,'cpu_seconds':time.process_time()-cpu})
                        sig=[(r['final'],r['stop'],[(a['position'],a['value']) for a in r['actions']]) for r in out]
                        if method in signatures:
                            assert sig==signatures[method],'Nonreproducible timed output'
                        signatures[method]=sig
                baseline_median=statistics.median(t['wall_seconds'] for t in entries['C64'])
                for method,times in entries.items():
                    med=statistics.median(t['wall_seconds'] for t in times)
                    measurements.append({'seed':seed,'batch':batch,'parse_cache':cache,'method':method,
                        'requests_per_repeat':len(balanced),'repeats':5,'measurements':times,
                        'median_wall_seconds':med,'median_ms_per_request':1000*med/len(balanced),
                        'C64_time_divided_by_method_time':baseline_median/med,
                        'same_complete_outputs_as_C64':sum(a==b for a,b in zip(signatures['C64'],signatures[method],strict=True)),
                        'outputs_repeatable':True})
                print('TIME',seed,batch,cache,{m:round(statistics.median(t['wall_seconds'] for t in ts),5) for m,ts in entries.items()},flush=True)
    write(root/'timing.json',{'public_requests':[r['id'] for r in balanced],
       'includes':['validation','parsing/graph lookup','encoding','neural computation where applicable','full output edit trace'],
       'excludes':['model loading','compilation','warmup','VM scoring','disk writes','network transfer'],
       'result_cache':False,'CPU_threads':1,'parallel_training_or_evaluation':False,
       'quality_preserving_speedup_claim':False,'results':measurements})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
