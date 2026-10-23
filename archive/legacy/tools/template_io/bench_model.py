from __future__ import annotations
import argparse,hashlib,json,random,statistics,sys,time,platform
from pathlib import Path
from template_io import ExactRequestCache,CacheContext

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.run));from predict_candidate import predict,common,runtime_setup
 runtime_setup();model_path=a.run/'runs/expanded-130363/model.pt';model=common.load_model(model_path)
 sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
 # Fingerprint all local runtime sources, not only top-level predictor.
 hh=hashlib.sha256()
 for path in sorted((a.run/'runtime').rglob('*.py')):hh.update(str(path.relative_to(a.run)).encode());hh.update(path.read_bytes())
 hh.update((a.run/'predict_candidate.py').read_bytes())
 import torch
 ctx=CacheContext('local-user',sha(model_path),hh.hexdigest(),decoding='argmax-fp32-cpu;torch='+torch.__version__+';python='+platform.python_version())
 inp=json.loads(a.inputs.read_text());queries=[(x['command'],bytes.fromhex(x['binary'])) for x in inp]
 def compute(q):
  b,t=predict(model,q[1],q[0]);return json.dumps({'binary':b.hex(),'trace':t},sort_keys=True,separators=(',',':')).encode()
 # Load and operation warm-up outside measurements; no result cache is warmed.
 for q in queries[:2]:compute(q)
 results=[];all_traces={}
 for repetitions in [1,2,10]:
  workload=queries*repetitions;random.Random(78311).shuffle(workload);un=[];ca=[];comparison=[]
  for run_index in range(3):
   outputs={};times={};cache=ExactRequestCache(max_entries=64)
   order=['uncached','cached'] if run_index%2==0 else ['cached','uncached']
   for mode in order:
    t=time.perf_counter();cpu=time.process_time()
    values=[compute(q) if mode=='uncached' else cache.get_or_compute(ctx,q[0],q[1],lambda q=q:compute(q)) for q in workload]
    times[mode]={'wall':time.perf_counter()-t,'cpu':time.process_time()-cpu};outputs[mode]=values
   if outputs['uncached']!=outputs['cached']:raise AssertionError('cache changed predictions')
   un.append(times['uncached']);ca.append(times['cached']);comparison.append({'hits':cache.hits,'misses':cache.misses,'accounted_cache_bytes':cache.bytes})
   for q,v in zip(workload,outputs['cached']):all_traces[hashlib.sha256(q[0].encode()+q[1]).hexdigest()]=json.loads(v)
  u=statistics.median(x['wall'] for x in un);c=statistics.median(x['wall'] for x in ca)
  result=dict(requests=len(workload),unique_requests=len(queries),repeat_share=1-1/repetitions,
   uncached_seconds=u,cached_cold_seconds=c,speedup=u/c,uncached_runs=un,cached_runs=ca,cache= comparison,all_byte_and_trace_matches=True)
  results.append(result);print(json.dumps(result),flush=True)
 a.out.write_text(json.dumps({'model_sha256':ctx.model_sha256,'runtime_sha256':ctx.runtime_sha256,'model_parameters':sum(p.numel() for p in model.parameters()),'rounds':64,
  'torch':torch.__version__,'results':results,'model_accuracy_tested':False,'training_steps':0,
  'scope':'Exact repeated frozen requests only; controlled workload, not observed production traffic. Cold result cache includes misses. Inputs are from training, not new capability tests.'},indent=2)+'\n')
 a.out.with_name('model_traces.json').write_text(json.dumps(all_traces,indent=2)+'\n')
if __name__=='__main__':main()
