"""Persistent local JSONL interface around unchanged LightNest C64 weights.

One line in: {"request": "...", "program_hex": "..."}; one JSON line out.
No model labels, execution outcomes, or ground-truth patches are loaded.
The exact cache is only used while this process holds fixed weights.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from pathlib import Path
from template_io import CacheContext,ExactRequestCache
from topology_cache import TopologyCache

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--namespace',required=True);p.add_argument('--cache-entries',type=int,default=512);p.add_argument('--cache-bytes',type=int,default=16<<20)
    a=p.parse_args();sys.path.insert(0,str(a.run_dir));from predict_candidate import predict,common,runtime_setup
    runtime_setup();model=common.load_model(a.model)
    hh=hashlib.sha256()
    for root in [a.run_dir/'runtime',Path(__file__).parent]:
        for f in sorted(root.rglob('*.py')):hh.update(str(f.relative_to(root)).encode());hh.update(f.read_bytes())
    hh.update((a.run_dir/'predict_candidate.py').read_bytes())
    import torch
    context=CacheContext(a.namespace,hashlib.sha256(a.model.read_bytes()).hexdigest(),hh.hexdigest(),
        decoding=f'argmax-fp32-cpu;torch={torch.__version__};python={platform.python_version()}')
    common.graph=TopologyCache(common.graph.__wrapped__,1024)
    cache=ExactRequestCache(a.cache_entries,a.cache_bytes)
    for line in sys.stdin:
        try:
            if len(line)>16384:raise ValueError('request line too long')
            obj=json.loads(line)
            if set(obj)!= {'request','program_hex'}:raise ValueError('only request and program_hex are accepted')
            command=obj['request'];hexcode=obj['program_hex']
            if not isinstance(command,str) or not 0<len(command.encode())<=model.config.command_capacity:raise ValueError('request length')
            if not isinstance(hexcode,str) or len(hexcode)>2048:raise ValueError('binary length')
            binary=bytes.fromhex(hexcode)
            def calculate():
                output,trace=predict(model,binary,command)
                return json.dumps({'program_hex':output.hex(),'trace':trace},sort_keys=True,separators=(',',':')).encode()
            hits=cache.hits;result=json.loads(cache.get_or_compute(context,command,binary,calculate))
            result['exact_cache_hit']=cache.hits>hits;result['model_sha256']=context.model_sha256
            print(json.dumps(result,separators=(',',':')),flush=True)
        except Exception as exc:print(json.dumps({'error':type(exc).__name__,'detail':str(exc)}),flush=True)
if __name__=='__main__':main()
