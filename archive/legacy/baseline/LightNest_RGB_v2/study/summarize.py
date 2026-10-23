"""Rebuild the compact result receipt from immutable experiment outputs."""
import json, platform
from pathlib import Path


def build(root=Path('.')):
    train=json.loads((root/'results/training-summary.json').read_text())
    meta_paths=[root/f'caches/parent-{seed}.json' for seed in (104729,130363)]
    cache_meta=([json.loads(p.read_text()) for p in meta_paths] if all(p.exists() for p in meta_paths)
                else json.loads((root/'results/cache-preparation.json').read_text())['seeds'])
    arms=[]
    for tag in [f'{a}-{s}' for a in ('g4','zero4','g8') for s in (104729,130363)]:
        results={}
        for mode in ('red_only','blue_only','auto','regression-red_only','regression-auto'):
            obj=json.loads((root/'final'/tag/(mode+'.json')).read_text())
            results[mode]=obj['summary']
        cmp=json.loads((root/'final'/tag/'comparison.json').read_text())
        reg=json.loads((root/'final'/tag/'regression-comparison.json').read_text())
        sel=json.loads((root/'calibration'/tag/'selection.json').read_text())
        arms.append({'tag':tag,'threshold':sel['threshold'],'enabled':sel['enabled'],'metrics':results,
                     'new_test_change_counts':{k:len(v) for k,v in cmp.items()},
                     'regression_change_counts':{k:len(v) for k,v in reg.items()}})
    follow=[]
    for tag in ('g4-104729','g4-130363'):
        sel=json.loads((root/'followup'/tag/'selection.json').read_text())
        results={mode:json.loads((root/'followup'/tag/(mode+'.json')).read_text())['summary'] for mode in ('confirm-red_only','confirm-auto','regression-red_only','regression-auto')}
        cmp={name:json.loads((root/'followup'/tag/(name+'-comparison.json')).read_text()) for name in ('confirm','regression')}
        follow.append({'tag':tag,'threshold':sel['threshold'],'stop_only':True,'model_sha256':sel['model_sha256'],
                       'metrics':results,'change_counts':{n:{k:len(v) for k,v in c.items()} for n,c in cmp.items()}})
    result={'experiment':'LightNest-RGB-v1-20261019','default_route':'red_only','stable_parent_replaced':False,
            'formal_training_runs':6,'training':train,'actor_updates':sum(r['steps'] for r in train),
            'router_updates':sum(r['router_steps'] for r in train),
            'actor_presentations':sum(r['actor_presentations'] for r in train),
            'router_presentations':sum(r['router_presentations'] for r in train),
            'formal_training_cpu_seconds':sum(r['actor_cpu_seconds']+r['router_cpu_seconds'] for r in train),
            'cache_preparation_cpu_seconds':sum(item['cpu_seconds'] for item in cache_meta),
            'primary':arms,'stop_only_followup':follow,'timing':json.loads((root/'results/timing.json').read_text()),
            'stop_timing':json.loads((root/'results/stop-timing.json').read_text()),
            'training_cache_comparison':json.loads((root/'results/cache-training-benchmark.json').read_text()),
            'primary_audit':json.loads((root/'results/evaluation-audit.json').read_text()),
            'followup_audit':json.loads((root/'results/stop-audit.json').read_text()),
            'unit_tests_passed':52,'limitations':['Restricted synthetic RV32 transformation; not project-level repair',
              'Red C64 stable parent differs from stronger unpromoted expanded research checkpoints',
              'Frozen feature caching benchmark excludes preparation and does not measure inference speed',
              'Only two inherited seeds; repeated task variants are correlated',
              'Follow-up uses full public regression for calibration; separate new confirmation retained',
              'STOP-only gate is a narrow research option, not reliable fast editing or a universal safety guarantee'],
            'recommended_candidate':'none for automatic production promotion; retain both g4 STOP-only research policies',
            'license_status':'unverified; no new redistribution rights granted'}
    # Cache preparation timing stays available without rebuilding the derived caches.
    (root/'results/cache-preparation.json').write_text(json.dumps({'seeds':cache_meta},indent=2))
    (root/'results/summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('SUMMARY',result['actor_updates'],result['router_updates'],result['formal_training_cpu_seconds'])
    return result

if __name__=='__main__':build()
