"""Five repeated live end-to-end proposal timings; no result cache or VM grading."""
import json,time,statistics,hashlib
from pathlib import Path
from collections import defaultdict
from experiment.run_study import ROOT,read,write,run_mode,load_rgb,configure_torch,equivalence

def digest(obj):
 return hashlib.sha256(json.dumps([{'id':r['id'],'final':r['final'],'actions':[(a['position'],a['value'],a['stop']) for a in r['actions']]} for r in obj['records']],sort_keys=True).encode()).hexdigest()

def main():
 configure_torch();rows=read(ROOT/'baseline_data/calibration.jsonl');by=defaultdict(list)
 for r in rows:by[r['family']].append(r)
 workloads={'simple':by['simple'][:32],'balanced':[r for fam in sorted(by) for r in by[fam][:4]]}
 assert len(workloads['balanced'])==32
 modes=['red','v1_stop','risk_full','verified'];records=[]
 write(ROOT/'results/benchmark-plan.json',{'modes':modes,'seeds':[104729,130363],'repeats':5,'batch_sizes':[1,16],
   'ids':{k:[r['id'] for r in v] for k,v in workloads.items()},'timed':'live full proposal incl scratch draft, route, verifier and fallback; excludes model load, file IO, VM and parser warmup','no_result_cache':True})
 for seed in (104729,130363):
  m=load_rgb(ROOT/f'models/rgb-{seed}.pt')
  for workload,eps in workloads.items():
   for batch in (1,16):
    refs={};samples=defaultdict(list);counts={}
    # Warm up each exact workload in each mode before repeat timing.
    for mode in modes:
     o=run_mode(m,eps,seed,mode,batch=batch,score=False);refs[mode]=digest(o)
    assert refs['verified']==refs['red'],'verified non-equivalence in timing warmup'
    for repeat in range(5):
     order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
     for mode in order:
      o=run_mode(m,eps,seed,mode,batch=batch,score=False)
      assert digest(o)==refs[mode],(seed,workload,batch,mode,'non-deterministic output')
      samples[mode].append(o['wall_seconds']);counts[mode]=o['compute']
    red=statistics.median(samples['red'])
    for mode in modes:
     med=statistics.median(samples[mode]);records.append({'seed':seed,'workload':workload,'batch':batch,'mode':mode,'rows':len(eps),
        'wall_seconds':samples[mode],'median':med,'speedup':red/med,'output_digest':refs[mode],'compute_per_run':counts[mode]})
    print('TIMING',seed,workload,batch,{k:round(red/statistics.median(v),3) for k,v in samples.items()},flush=True)
 write(ROOT/'results/timing-v2.json',{'records':records,'no_concurrent_training':True,'threads':1,'repeats':5,
   'mode_specific_proposals_can_differ':'only verified is intended to equal red; risk may save time by correcting over-editing or causing early termination, so its speed is not a pure model-compute speedup'})
if __name__=='__main__':main()
