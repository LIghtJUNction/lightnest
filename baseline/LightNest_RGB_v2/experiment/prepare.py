"""Create fresh group-separated holdouts before running any new model test."""
from pathlib import Path
import gzip,hashlib,json,random,zipfile,time
from collections import Counter
from lightnest_data.core import neutral_fingerprint, validate_program, file_sha
from lightnest_data.vm import execute_a, execute_b
from lightnest_data.generator import FAMILIES,generate_pair
from study.prepare import simple_pair,write_rows

def main():
 root=Path(__file__).resolve().parents[1]; start=time.process_time();known=set();exact=set();inputs_seen=0
 # Read metadata only from expanded release: no model evaluation of its labels.
 with zipfile.ZipFile('/mnt/data/LightNest_scale_v2_10k_dataset.zip') as z:
  names=[n for n in z.namelist() if n.endswith('.jsonl.gz') and '/index/' in n]
  if not names:
   names=[n for n in z.namelist() if n.endswith('.jsonl.gz') and 'index' in n]
  print('index_shards',len(names),flush=True)
  for n in names:
   with gzip.GzipFile(fileobj=z.open(n)) as f:
    for line in f:
     v=json.loads(line);known.add(v['neutral_sha256']);exact.update((v['binary_sha256'],v['target_sha256']));inputs_seen+=1
 if not names:raise RuntimeError('expanded indexes not located')
 oldroot=Path('/mnt/data/rgb_v2_work/LightNest_RGB_v1'); old_paths=list((oldroot/'data').glob('*.jsonl'))+[oldroot/'followup/confirm.jsonl']
 for p in old_paths:
  for line in p.read_text().splitlines():
   r=json.loads(line)
   for k in ('binary','target'):
    b=bytes.fromhex(r[k]);known.add(neutral_fingerprint(b));exact.add(hashlib.sha256(b).hexdigest())
 protocol=json.loads((root/'protocol.json').read_text());summ={};calls=0
 rr=random.Random(202610192);cases=[[0,0,0],[1,2,3],[-1,2,-3],[2,5,3],[2**31-1,1,-1],[-2**31,-1,1],[7,7,0],[-13,13,5]]+[[rr.randint(-100000,100000) for _ in range(3)] for i in range(8)]
 for role,cfg in protocol['fresh_tests'].items():
  if role not in ('normal','stress'):continue
  rows=[];collisions=0
  for fi,fam in enumerate(protocol['fresh_tests']['families']):
   got=0;i=0
   while got<cfg['pairs_per_family']:
    seed=cfg['seed']+fi*100000+i;i+=1
    es=simple_pair(seed,got%2,'rgb-v2/'+role) if fam=='simple' else generate_pair(fam,seed,got%2,tuple(cfg['chain']),'rgb-v2/'+role)
    fs={neutral_fingerprint(bytes.fromhex(e[k])) for e in es for k in ('binary','target')}
    xs={hashlib.sha256(bytes.fromhex(e[k])).hexdigest() for e in es for k in ('binary','target')}
    if fs&known or xs&exact:collisions+=1;continue
    known.update(fs);exact.update(xs)
    for e in es:
     e.pop('source',None);e['cases']=[]
     for field in ('binary','target'):validate_program(bytes.fromhex(e[field]))
     for x in cases:
      for field in ('binary','target'):
       a=execute_a(bytes.fromhex(e[field]),x);b=execute_b(bytes.fromhex(e[field]),x);calls+=2
       if a!=b:raise AssertionError((role,e['id'],field,x,a,b))
       if field=='target':e['cases'].append([x,a])
    rows.extend(es);got+=1
  write_rows(root/'data'/f'{role}.jsonl',rows)
  summ[role]={'rows':len(rows),'groups':len({r['pair_id'] for r in rows}),'collisions':collisions,
    'families':dict(Counter(r['family'] for r in rows)),'sha256':file_sha(root/'data'/f'{role}.jsonl'),
    'min_bytes':min(len(r['binary'])//2 for r in rows),'max_bytes':max(len(r['binary'])//2 for r in rows),'model_queries':0}
 (root/'results/construction.json').write_text(json.dumps({'expanded_index_rows':inputs_seen,'new':summ,'dual_vm_calls':calls,'cpu_seconds':time.process_time()-start,'old_sources':{str(p.relative_to(oldroot)):file_sha(p) for p in old_paths}},indent=2))
 print(json.dumps(summ),flush=True)
if __name__=='__main__':main()
