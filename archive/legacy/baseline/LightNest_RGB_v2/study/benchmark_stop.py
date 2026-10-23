"""Measure corrected STOP-only gate; policy frozen before timing, cache off."""
import json,time,statistics
from pathlib import Path
from rgb.runtime import configure_torch
from rgb.stop_gate import load_stop_only,StopOnlyPolicy
from .cache import read_rows
from .evaluate import Live,rollout

def main(root=Path('.')):
    configure_torch();rows=read_rows(root/'data/calibration.jsonl');families=list(dict.fromkeys(e['family'] for e in rows))
    workloads={'simple64':rows[:64],'balanced64':[e for fam in families for e in [r for r in rows if r['family']==fam][:8]]}
    out=[]
    for tag in ('g4-104729','g4-130363'):
        m=load_stop_only(root/'runs'/tag/'model.pt');sel=json.loads((root/'followup'/tag/'selection.json').read_text());m.policy=StopOnlyPolicy(sel['enabled'],sel['threshold'])
        for name,rr in workloads.items():
            refs={mode:rollout(Live(m,mode),rr,score=False) for mode in ('red_only','auto')};times={mode:[] for mode in refs}
            for i in range(5):
                for mode in (('auto','red_only') if i%2 else ('red_only','auto')):
                    c=time.process_time();w=time.perf_counter();o=rollout(Live(m,mode),rr,score=False);dt=time.perf_counter()-w;dc=time.process_time()-c
                    assert o==refs[mode];times[mode].append({'cpu':dc,'wall':dt})
            base=statistics.median(t['wall'] for t in times['red_only']);new=statistics.median(t['wall'] for t in times['auto'])
            entry={'tag':tag,'workload':name,'episodes':len(rr),'times':times,'red_wall_median':base,'auto_wall_median':new,'speedup':base/new,
                   'blue_actions':sum(a['blue'] for r in refs['auto'] for a in r['actions']),
                   'blue_edit_actions':sum(a['blue'] and not a['stop'] for r in refs['auto'] for a in r['actions']),
                   'cache':False,'vm_scoring_timed':False,'parent_loading_timed':False,'static_parse_cache_warm':True}
            out.append(entry);print('STOP_TIMING',tag,name,base/new,flush=True)
    (root/'results/stop-timing.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
