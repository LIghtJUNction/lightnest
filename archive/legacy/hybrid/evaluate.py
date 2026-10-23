"""Frozen endpoint evaluation. Inference never sees a target or VM result."""
from __future__ import annotations
import argparse,json,time,hashlib
from datetime import datetime,timezone
from . import ROOT,BASE
from .model import *
from .prepare import read,write,sha
from rgb.scripted_core import compile_rgb
from study.evaluate import Live,rollout,summarize,compare

MODES=('parent64','sft64','sft','rl0','rlcost')
def build(seed,mode):
    if mode=='parent64':return Live(compile_rgb(load_rgb(BASE/f'models/rgb-{seed}.pt')),'red_only')
    m=load_actor(seed)
    if mode=='sft64':
        @torch.inference_mode()
        def fixed(rows):
            inputs,rel,ctx,h=m.start(rows);prev=0
            for depth in DEPTHS:
                h,g=m.block(h,inputs,rel,ctx,depth-prev,True);prev=depth
            sc,bi=m.head(h,ctx);out=[]
            for j in range(len(rows)):
                a=action(sc[j],bi[j],m.none_index)
                out.append(a|{'blue':False,'rounds':64,'text':encode_action(a),'mode_trace':[{'mode':'THINK' if d<64 else 'EMIT','depth':d} for d in DEPTHS]})
            return out
        return fixed
    p=load_policy(seed,mode)
    return lambda rows:m.predict(rows,p)
def freeze():
    for s in (104729,130363):
        if not (ROOT/f'results/train-complete-{s}.json').exists():raise ValueError('training unfinished')
    original=json.loads((ROOT/'source-freeze.json').read_text())
    for name,h in original['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('training source changed '+name)
    prot=json.loads((ROOT/'protocol.json').read_text())
    for name,h in prot['data'].items():assert sha(ROOT/'data'/name)==h
    names=list((ROOT/'models').glob('*.pt'))+list((ROOT/'hybrid').glob('*.py'))+[ROOT/'data/test.jsonl',ROOT/'protocol.json',BASE/'baseline_data/regression.jsonl']
    write(ROOT/'final-freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'test_model_queries':0,
             'files':{str(p.relative_to(ROOT)):sha(p) for p in names},'strategies':MODES})
def evaluate(seed):
    configure_torch();f=json.loads((ROOT/'final-freeze.json').read_text())
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    for split,path in [('test',ROOT/'data/test.jsonl'),('regression',BASE/'baseline_data/regression.jsonl')]:
        es=read(path);baseline=None
        for mode in MODES:
            pred=build(seed,mode);cpu=time.process_time();wall=time.perf_counter();rs=rollout(pred,es,batch=16)
            obj={'seed':seed,'split':split,'mode':mode,'records':rs,'summary':summarize(rs),'cpu_seconds_including_score':time.process_time()-cpu,'wall_seconds_including_score':time.perf_counter()-wall}
            if mode=='parent64':baseline=rs
            else:obj['comparison_to_parent64']=compare(baseline,rs)
            write(ROOT/f'results/final-{seed}-{split}-{mode}.json',obj)
            print('EVAL',seed,split,mode,obj['summary']['all'],flush=True)
    write(ROOT/f'results/evaluation-complete-{seed}.json',{'complete':True,'test_rows':256,'regression_rows':2372})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--seed',type=int,default=104729);a=p.parse_args()
    freeze() if a.phase=='freeze' else evaluate(a.seed)
