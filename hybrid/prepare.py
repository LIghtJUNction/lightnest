"""Freeze protocol and group-separated new tests before model training."""
import json,hashlib,zipfile,gzip,random,time
from collections import Counter
from . import ROOT,BASE
from lightnest_data.core import neutral_fingerprint
from lightnest_data.generator import generate_pair,FAMILIES
from lightnest_data.vm import execute_a,execute_b
from study.prepare import simple_pair

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    start=time.process_time();known=set();exact=set();checked=0
    manifest=json.loads((BASE/'FILES.sha256.json').read_text())['files']
    for n,v in manifest.items():
        assert sha(BASE/n)==v['sha256'];checked+=1
    # Use the already-designated TRAIN pool. Full v1 fit is extracted from an
    # unchanged archive; v2 router_fit is a disjoint TRAIN subset for mode/RL.
    with zipfile.ZipFile('/mnt/data/LightNest_RGB_v1_source_models_results.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('/data/fit.jsonl'))
        fit=[json.loads(l) for l in z.read(name).decode().splitlines()]
        historical=[]
        for n in z.namelist():
            if n.endswith('.jsonl') and ('/data/' in n or n.endswith('/followup/confirm.jsonl')):
                historical += [json.loads(l) for l in z.read(n).decode().splitlines()]
    mode=read(BASE/'baseline_data/router_fit.jsonl')
    assert not {r['pair_id'] for r in fit}&{r['pair_id'] for r in mode}
    rg=random.Random(190119);cases=[[0,0,0],[1,2,3],[-1,2,-3],[2,5,3],[2**31-1,1,-1],[-2**31,-1,1],[7,7,0],[-13,13,5]]
    cases += [[rg.randint(-100000,100000) for _ in range(3)] for _ in range(8)]
    # Only training labels are attached here; the policy interface excludes them.
    for r in mode:r['cases']=[[x,execute_b(bytes.fromhex(r['target']),x)] for x in cases]
    for name,rs in [('fit',fit),('mode_train',mode)]:
        with (ROOT/'data'/f'{name}.jsonl').open('x') as f:
            for r in rs:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with zipfile.ZipFile('/mnt/data/LightNest_scale_v2_10k_dataset.zip') as z:
        for n in z.namelist():
            if n.endswith('.jsonl.gz') and 'index' in n:
                with gzip.GzipFile(fileobj=z.open(n)) as f:
                    for l in f:
                        v=json.loads(l);known.add(v['neutral_sha256']);exact.update([v['binary_sha256'],v['target_sha256']])
    for p in list((BASE/'baseline_data').glob('*.jsonl'))+list((BASE/'data').glob('*.jsonl')):historical+=read(p)
    for r in historical:
        for k in ('binary','target'):
            b=bytes.fromhex(r[k]);known.add(neutral_fingerprint(b));exact.add(hashlib.sha256(b).hexdigest())
    families=('simple',)+FAMILIES+('branch_spill','loop_spill');out=[];collisions=0;vm=0
    for fi,fam in enumerate(families):
        got=0;i=0
        while got<8:
            sd=991910190+100000*fi+i;i+=1
            length=(3,8) if got<4 else (9,12)
            es=simple_pair(sd,got%2,'hybrid-v1') if fam=='simple' else generate_pair(fam,sd,got%2,length,'hybrid-v1')
            fs={neutral_fingerprint(bytes.fromhex(e[k])) for e in es for k in ('binary','target')}
            xs={hashlib.sha256(bytes.fromhex(e[k])).hexdigest() for e in es for k in ('binary','target')}
            if fs&known or xs&exact:collisions+=1;continue
            known.update(fs);exact.update(xs)
            for e in es:
                e.pop('source',None);e['cases']=[];e['difficulty']='normal' if got<4 else 'stress'
                for x in cases:
                    for key in ('binary','target'):
                        a=execute_a(bytes.fromhex(e[key]),x);b=execute_b(bytes.fromhex(e[key]),x);vm+=2
                        assert a==b
                        if key=='target':e['cases'].append([x,a])
            out+=es;got+=1
    with (ROOT/'data/test.jsonl').open('x') as f:
        for r in out:f.write(json.dumps(r,separators=(',',':'))+'\n')
    from datetime import datetime,timezone
    protocol={'id':'LightNest-Hybrid-v1','created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[104729,130363],
       'depths':[4,8,16,32,64], 'sft':{'steps':192,'batch':8,'lr':0.0003,'weight_decay':0.01,'clip':1.,'latent_mse_weight':0.02,
          'sampling':'4 legacy plus 4 hard, equal EDIT/STOP in each; same stream per seed','frozen':'C64 backbone and inherited green','trainable':'residual link and copied action head','self_conditioned':True},
       'mode_sft':{'steps':256,'batch':32,'lr':0.001,'target':'EMIT if current greedy action matches any remaining legal edit, or correct STOP; otherwise THINK'},
       'rl':{'updates':96,'tasks_per_update':4,'rollouts_per_task':4,'lr':0.0003,'entropy':0.005,'kl_to_sft':0.01,'cost_weights':[0.,0.1],
          'reward':'1 for exact+STOP+finite execution, otherwise -1; additional -1 for editing no-op; minus cost_weight*rounds/(9*64)',
          'policy_gradient':'on-policy REINFORCE, leave-one-out group baseline, only sampled mode decisions','frozen':'actor, link, green, recurrent core',
          'replay':'recorded frozen latent-derived features reused without recomputing or relabeling as sampled tokens'},
       'evaluation':{'test_rows':len(out),'full_public_regression':2372,'max_edits':8,'final_STOP_opportunity':True,
          'modes':['parent64','sft64','sft','rl0','rlcost'],'selection':'report all fixed endpoints, no test or seed selection',
          'promotion':'no wrong no-op; no loss of parent successes on test and full regression; useful improvement required; default remains parent'},
       'scope':['bounded RV32 actions serialized as text; not a general language decoder','existing generator families, not real project bugs',
                'reset working state after any code edit; preserve it across THINK steps','no external agent, API training or native binary execution',
                'adaptation of HybridAgent ideas, not paper reproduction'],
       'data':{p.name:sha(p) for p in (ROOT/'data').glob('*.jsonl')},
       'baseline_checkpoints':{str(s):sha(BASE/f'models/rgb-{s}.pt') for s in (104729,130363)}}
    write(ROOT/'protocol.json',protocol)
    write(ROOT/'results/construction.json',{'baseline_members_verified':checked,'fit_rows':len(fit),'mode_rows':len(mode),'fit_groups':len({r['pair_id'] for r in fit}),
       'mode_groups':len({r['pair_id'] for r in mode}),'test_groups':len({r['pair_id'] for r in out}),'test_families':dict(Counter(r['family'] for r in out)),
       'collisions_rejected':collisions,'validation_vm_calls':vm,'cpu_seconds':time.process_time()-start,'test_model_queries':0})
    print('PREPARED',len(fit),len(mode),len(out),flush=True)
if __name__=='__main__':main()
