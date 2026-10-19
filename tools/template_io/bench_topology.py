from __future__ import annotations
import argparse,hashlib,json,random,statistics,sys,time
from functools import lru_cache
from pathlib import Path
from template_io import unpack_core
from topology_cache import TopologyCache,topology_bytes

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--core',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.run/'src'));from shared import common,runtime_setup
 runtime_setup();raw=common.graph.__wrapped__;fast=TopologyCache(raw,1024);rows=list(unpack_core(a.core.read_bytes()));data=[r.binary for r in rows]
 begin=time.perf_counter();checks=0
 for b in data:
  if raw(b)!=fast(b):raise AssertionError('changed static graph')
  checks+=1
 print('all graphs checked',checks,flush=True)
 validation_seconds=time.perf_counter()-begin
 # First 4096 rows, two orders. This is parser-only, not neural speedup.
 sample=data[:4096];orders={'group_order':sample,'shuffled_order':random.Random(48203).sample(sample,len(sample))}
 stats={}
 for name,seq in orders.items():
  ref=lru_cache(maxsize=1024)(raw);opt=TopologyCache(raw,1024);times={'exact_key':[],'topology_key':[]};hits={}
  for repeat in range(3):
   for key,fn in ([('exact_key',ref),('topology_key',opt)] if repeat%2==0 else [('topology_key',opt),('exact_key',ref)]):
    fn.cache_clear();t=time.perf_counter();values=[fn(x) for x in seq];times[key].append(time.perf_counter()-t);hits[key]=fn.cache_info()._asdict()
  stats[name]=dict(records=len(seq),times=times,cache=hits,speedup=statistics.median(times['exact_key'])/statistics.median(times['topology_key']))
 a.out.write_text(json.dumps(dict(records_verified=checks,topology_mismatches=0,full_verification_seconds=validation_seconds,
  input_only=True,model_tokens_modified=False,model_states_reused=False,source_sha256=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
  unique_full_binary=len(set(data)),unique_topology_keys=len({topology_bytes(x) for x in data}),parser_benchmarks=stats,
  limitation='No general alias analysis or unknown instruction support added; parser-only speed is not end-to-end neural speed.'),indent=2)+'\n')
 print(a.out.read_text(),flush=True)
if __name__=='__main__':main()
