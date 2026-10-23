"""Cache-free full-proposal-sequence latency. No target execution in timed loop.

Warm frozen models and static graph parse cache. Exact result/observation cache
is OFF. Wall and process CPU measured. Workload is explicitly identified and
includes failures; speed is never promoted without preservation checks.
"""
import json,time,statistics
from pathlib import Path
from rgb.runtime import configure_torch
from rgb.network import load_rgb,RoutePolicy
from lightnest_data.core import file_sha
from .cache import read_rows
from .evaluate import Live,rollout

def main(root=Path('.')):
    configure_torch();eps=read_rows(root/'data/calibration.jsonl')
    simple=eps[:64]
    families=list(dict.fromkeys(e['family'] for e in eps))
    balanced=[e for family in families for e in [x for x in eps if x['family']==family][:8]]
    spec={'primary':{'ids':[e['id'] for e in simple],'selection':'first64 calibration in original order; all simple'},
          'balanced_diagnostic':{'ids':[e['id'] for e in balanced],'selection':'first8 per family, declared before timing'},
          'repetitions':5,'batch':16,'warmup':True,'result_cache':False,'includes_model_loading':False,
          'includes_vm_scoring':False,'static_graph_cache':'warm bounded512','order':'rotating interleaving'}
    (root/'results/timing-protocol.json').write_text(json.dumps(spec,indent=2))
    allresults=[]
    for fp in sorted((root/'runs').glob('*/model.pt')):
        tag=fp.parent.name;m=load_rgb(fp);s=json.loads((root/'calibration'/tag/'selection.json').read_text());m.policy=RoutePolicy(s['enabled'],s['threshold'])
        for label,rows in [('primary',simple),('balanced_diagnostic',balanced)]:
            modes=['red_only','blue_only','auto'];ref={}
            for mode in modes:ref[mode]=rollout(Live(m,mode),rows,score=False)
            times={mode:[] for mode in modes}
            for i in range(5):
                for mode in modes[i%3:]+modes[:i%3]:
                    cpu=time.process_time();wall=time.perf_counter();out=rollout(Live(m,mode),rows,score=False)
                    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
                    assert out==ref[mode]
                    times[mode].append({'wall_seconds':wall,'cpu_seconds':cpu})
            base=statistics.median(x['wall_seconds'] for x in times['red_only'])
            stats={mode:{'trials':trials,'wall_median':statistics.median(x['wall_seconds'] for x in trials),
                         'cpu_median':statistics.median(x['cpu_seconds'] for x in trials),
                         'speedup_over_red':base/statistics.median(x['wall_seconds'] for x in trials),
                         'actions':sum(len(r['actions']) for r in ref[mode]),
                         'blue_edit_actions':sum(a['blue'] and not a['stop'] for r in ref[mode] for a in r['actions']),
                         'blue_stop_actions':sum(a['blue'] and a['stop'] for r in ref[mode] for a in r['actions']),
                         'round_examples':sum(a['rounds'] for r in ref[mode] for a in r['actions'])} for mode,trials in times.items()}
            allresults.append({'tag':tag,'workload':label,'episodes':len(rows),'model_sha256':file_sha(fp),'metrics':stats})
            print('TIMING',tag,label,'auto_speedup',stats['auto']['speedup_over_red'],flush=True)
    (root/'results/timing.json').write_text(json.dumps(allresults,indent=2))
if __name__=='__main__':main()
