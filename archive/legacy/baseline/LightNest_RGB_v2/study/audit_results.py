"""Independent read-only replay/count implementation for all final records.

Does NOT import study.evaluate's scoring or metrics functions. Reconstructs
candidate bytes from recorded actions, re-executes the second VM, recomputes
success and false edits and compares red/auto IDs. It is a same-session script,
not an external organization or an independent research team.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from lightnest_data.core import file_sha
from lightnest_data.vm import execute_b

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]

def main(root=Path('.')):
    frozen=json.loads((root/'final-freeze.json').read_text())
    assert file_sha(root/'study/evaluate.py')==frozen['evaluator_sha256']
    for p,h in frozen['inference_source'].items():assert file_sha(root/p)==h
    checks=[];calls=0;sequences=0
    for tag,hashval in frozen['models'].items():
        assert file_sha(root/'runs'/tag/'model.pt')==hashval
        assert file_sha(root/'calibration'/tag/'selection.json')==frozen['selections'][tag]
        modes={}
        for f in sorted((root/'final'/tag).glob('*.json')):
            payload=json.loads(f.read_text())
            if 'records' not in payload:continue
            name='regression.jsonl' if f.name.startswith('regression-') else 'test.jsonl'
            eps={e['id']:e for e in read(root/'data'/name)};rs=payload['records'];assert len(eps)==len(rs)
            assert len({r['id'] for r in rs})==len(rs)
            count=Counter();pairs=defaultdict(list)
            for r in rs:
                e=eps[r['id']];code=bytearray.fromhex(e['binary']);stop=False;proposals=0;cost=0
                for idx,a in enumerate(r['actions']):
                    assert a['step']==idx;cost+=a['rounds']
                    if a['stop']:stop=True;assert idx==len(r['actions'])-1;break
                    proposals+=1
                    if idx==8:continue
                    p=a['position']
                    if not 0<=p<len(code):continue
                    assert a['before']==code[p];code[p]=a['value']
                assert bytes(code).hex()==r['final'];assert cost==r['round_examples']
                ok=True;numeric=[]
                for x,y in e['cases']:
                    calls+=1
                    try:v=execute_b(bytes(code),x)
                    except (ValueError,IndexError):v=None
                    numeric.append(v);ok &= v==y
                need=bool(e['edits']);strict=bool(bytes(code).hex()==e['target'] and stop and ok and (need or proposals==0))
                false=(not need and proposals>0)
                assert strict==r['strict'];assert false==r['false_edit'];assert ok==r['numeric'];assert stop==r['explicit_stop'];assert proposals==r['proposed']
                count['total']+=1;count['edits']+=need;count['noops']+=not need;count['strict_edits']+=need and strict
                count['strict_total']+=strict;count['false_edits']+=false;count['blue_episodes']+=r['used_blue'];count['round_examples']+=cost
                if need and e.get('lane') is not None:pairs[e['pair_id']].append(strict)
                sequences+=1
            pp=[v for v in pairs.values() if len(v)==2];count['pairs']=len(pp);count['pairs_both']=sum(all(p) for p in pp)
            for k,v in count.items():assert v==payload['summary']['all'][k],(tag,f.name,k)
            modes[f.stem]={r['id']:r for r in rs};checks.append({'tag':tag,'file':f.name,'sequences':len(rs),'replayed':True})
        for prefix,cmpfile in [('', 'comparison.json'),('regression-', 'regression-comparison.json')]:
            b=modes[prefix+'red_only'];c=modes[prefix+'auto'];cmp=json.loads((root/'final'/tag/cmpfile).read_text())
            lost={k for k,r in c.items() if b[k]['strict'] and not r['strict']}
            added={k for k,r in c.items() if not b[k]['strict'] and r['strict']}
            mis={k for k,r in c.items() if not b[k]['false_edit'] and r['false_edit']}
            assert lost==set(cmp['lost_success_ids']);assert added==set(cmp['added_success_ids']);assert mis==set(cmp['added_false_edit_ids'])
    report={'passed':True,'final_record_sequences':sequences,'second_vm_reexecution_calls':calls,'files':checks,
            'evaluator_and_network_unchanged':True,'models_and_policies_unchanged':True,
            'independence_scope':'different same-session metric code + supplied second VM; not external audit'}
    (root/'results/evaluation-audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_PASS',sequences,calls,flush=True)
if __name__=='__main__':main()
