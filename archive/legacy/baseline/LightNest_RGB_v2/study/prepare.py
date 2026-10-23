"""Build bounded, source-traceable training pools and unseen grouped tests.

Only the old dataset's TRAIN split supplies learning targets. All roles of a
construction group stay together. New tests are same-generator cases, not new
algorithms or real project patches. No model is loaded by this script.
"""
from __future__ import annotations
import argparse, gzip, hashlib, heapq, json, random
from collections import Counter
from pathlib import Path
from lightnest_data.core import file_sha, neutral_fingerprint, validate_program
from lightnest_data.generator import generate_pair, FAMILIES
from lightnest_data.vm import assemble, execute_a, execute_b


def write_rows(path, rows):
    with Path(path).open('x') as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True, separators=(',',':'))+'\n')


def simple_pair(seed, direction, prefix):
    rng = random.Random(seed)
    regs = rng.sample([x for x in range(1,32) if x != 10],10)
    x,y,a,b,ac,bc,dead,noise,*_ = regs
    op='add' if direction==0 else 'sub'
    lines=[f'lw x{x}, 0(x0)',f'lw x{y}, 4(x0)']
    offsets=[]
    for r,consumer in ((a,ac),(b,bc)):
        lines += [f'addi x{noise}, x0, {rng.randint(-1500,1500)}' for _ in range(rng.randint(0,3))]
        offsets.append(4*len(lines)+3)
        lines.append(f'{op} x{r}, x{x}, x{y}')
        lines.append(f'addi x{consumer}, x{r}, {rng.randint(-10,10)}')
        for _ in range(rng.randint(0,3)):
            lines.append(f'addi x{consumer}, x{consumer}, 0')
        lines.append(f'addi x{noise}, x0, {rng.randint(-1500,1500)}')
    lines.append(f'{op} x{dead}, x{x}, x{y}')
    rows=[];gid=f'{prefix}/simple/{seed}'
    for lane,consumer in enumerate((ac,bc)):
        source='\n'.join(lines+[f'addi x10, x{consumer}, 0','ebreak'])+'\n'
        raw=assemble(source);target=bytearray(raw);target[offsets[lane]] ^= 64
        for noop in (False,True):
            code=bytes(target) if noop else raw
            rows.append(dict(id=f'{gid}/{lane}/{"noop" if noop else "edit"}',pair_id=gid,family='simple',lane=lane,
                direction=direction,command=f'change every {op} that contributes to the output to {"sub" if direction==0 else "add"}',
                binary=code.hex(),target=bytes(target).hex(),edits=[] if noop else [dict(offset=offsets[lane],value=target[offsets[lane]])]))
    return rows


def prepare(root, out):
    out.mkdir(parents=True,exist_ok=False)
    m=json.loads((root/'manifest.json').read_text());buckets={};counts=Counter();known=set();known_exact=set();source_hashes={}
    groups={'fit':set(),'router_fit':set()}
    for s in m['shards']:
        for k in ('inputs','labels','index'):
            p=root/s[k]['path'];assert file_sha(p)==s[k]['sha256'];source_hashes[s[k]['path']]=s[k]['sha256']
        with gzip.open(root/s['inputs']['path'],'rt') as fi,gzip.open(root/s['labels']['path'],'rt') as fl,gzip.open(root/s['index']['path'],'rt') as fx:
            for li,ll,lx in zip(fi,fl,fx,strict=True):
                inp,lab,idx=map(json.loads,(li,ll,lx))
                assert inp['record_id']==lab['record_id']==idx['record_id']
                known.add(idx['neutral_sha256']);known_exact.update((idx['binary_sha256'],idx['target_sha256']))
                if s['split']!='train':continue
                fam=next((x for x in idx['families'] if x in FAMILIES),'legacy')
                role='router_fit' if int(idx['group_id'][:16],16)%5==0 else 'fit'
                cap=(128 if fam=='legacy' else 64) if role=='router_fit' else (512 if fam=='legacy' else 256)
                key=(role,fam,lab['action']);counts[key]+=1
                row=dict(id=inp['record_id'],pair_id=idx['group_id'],family=fam,command=inp['request'],binary=inp['program_hex'],
                         target=lab['target_hex'],edits=[dict(offset=e['offset'],value=e['after']) for e in lab['edits']])
                priority=int(hashlib.sha256((role+row['id']).encode()).hexdigest(),16)
                h=buckets.setdefault(key,[])
                if len(h)<cap:heapq.heappush(h,(-priority,row['id'],row))
                elif priority < -h[0][0]:heapq.heapreplace(h,(-priority,row['id'],row))
    summary={}
    for role in ('fit','router_fit'):
        rows=sorted([v[2] for k,h in buckets.items() if k[0]==role for v in h],key=lambda r:r['id'])
        groups[role]={r['pair_id'] for r in rows};write_rows(out/(role+'.jsonl'),rows)
        summary[role]={'rows':len(rows),'groups':len(groups[role]),'buckets':{str(k):len(h) for k,h in buckets.items() if k[0]==role}}
    assert not groups['fit']&groups['router_fit']
    tests=[[0,0,0],[1,2,3],[-1,2,-3],[2,5,3],[2147483647,1,-1],[-2147483648,-1,1],[7,7,0],[-13,13,5]]
    rng=random.Random(2026191000)
    tests += [[rng.randint(-100000,100000) for _ in range(3)] for _ in range(8)]
    vm_calls=0
    for role,seed0,count,simple_count in [('calibration',831019001,8,32),('test',941019001,16,64)]:
        rows=[];excluded=0
        for fi,fam in enumerate(('simple',)+FAMILIES+('branch_spill','loop_spill')):
            count_here=simple_count if fam=='simple' else count;accepted=0;i=0
            while accepted<count_here:
                seed=seed0+fi*100000+i;i+=1
                rr=simple_pair(seed,accepted%2,role) if fam=='simple' else generate_pair(fam,seed,accepted%2,(3,8),role)
                fingerprints={neutral_fingerprint(bytes.fromhex(r[k])) for r in rr for k in ('binary','target')}
                exact={hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in rr for k in ('binary','target')}
                if fingerprints&known or exact&known_exact:excluded+=1;continue
                known.update(fingerprints);known_exact.update(exact)
                for r in rr:
                    r.pop('source',None);r['cases']=[]
                    for k in ('binary','target'):validate_program(bytes.fromhex(r[k]))
                    for inp in tests:
                        for field in ('binary','target'):
                            x=execute_a(bytes.fromhex(r[field]),inp);y=execute_b(bytes.fromhex(r[field]),inp);vm_calls+=2
                            assert x==y
                            if field=='target':r['cases'].append([inp,x])
                rows+=rr;accepted+=1
        write_rows(out/(role+'.jsonl'),rows)
        summary[role]={'rows':len(rows),'groups':len(set(r['pair_id'] for r in rows)),'families':dict(Counter(r['family'] for r in rows)),
                       'excluded_collisions':excluded,'model_queries':0,'seed':seed0}
    # Existing public regression remains public, never label it as unseen.
    rows=[]
    for s in m['shards']:
        if s['split']!='regression_public':continue
        with gzip.open(root/s['inputs']['path'],'rt') as fi,gzip.open(root/s['labels']['path'],'rt') as fl,gzip.open(root/s['index']['path'],'rt') as fx:
            for li,ll,lx in zip(fi,fl,fx,strict=True):
                inp,lab,idx=map(json.loads,(li,ll,lx)); cases=[]
                for e in lab['execution']:
                    cs=json.loads((root/'cases'/(e['case_suite_id']+'.json')).read_text())
                    # Input suite names are part of the fixed source contract.
                    xs=cs['inputs']
                    cases.extend(list(zip(xs,e['expected_outputs'],strict=True)))
                rows.append(dict(id=inp['record_id'],pair_id=idx['group_id'],family=idx['families'][0],lane=None,
                    command=inp['request'],binary=inp['program_hex'],target=lab['target_hex'],cases=cases,
                    edits=[dict(offset=e['offset'],value=e['after']) for e in lab['edits']]))
    write_rows(out/'regression.jsonl',rows)
    summary['regression']={'rows':len(rows),'status':'previously_exposed_only'}
    summary['source_content_id']=m['content_id'];summary['source_manifest_sha256']=file_sha(root/'manifest.json')
    summary['source_shards']=source_hashes;summary['dual_vm_construction_calls']=vm_calls
    summary['files']={p.name:file_sha(p) for p in out.glob('*.jsonl')}
    summary['limitations']=['same generators and shared templates','source-group split, not semantic proof','source licenses unverified',
                           'no new test in fit or router_fit; calibration only tunes policies','bounded subset of train pool selected by hash']
    (out/'construction.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_shards'},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();prepare(a.data,a.out)
