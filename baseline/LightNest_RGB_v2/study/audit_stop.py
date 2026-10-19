"""Read-only independent replay for the separately frozen STOP-only follow-up.

No imports from the evaluator or learner. Counts saved actions independently
and runs the supplied second interpreter. Not an external third-party audit.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from lightnest_data.core import file_sha
from lightnest_data.vm import execute_b


def main(root=Path('.')):
    freeze=json.loads((root/'followup/final-freeze.json').read_text())
    protocol=json.loads((root/'followup/protocol.json').read_text())
    assert file_sha(root/'followup/protocol.json')==freeze['protocol_sha256']
    assert file_sha(root/'followup/confirm.jsonl')==freeze['test_sha256']
    for name,h in protocol['source_sha256'].items():
        assert file_sha(root/name)==h
    calls=sequences=0;files=[];comparisons=[]
    for tag,h in freeze['models'].items():
        assert file_sha(root/'runs'/tag/'model.pt')==h
        assert file_sha(root/'followup'/tag/'selection.json')==freeze['selections'][tag]
        chosen=json.loads((root/'followup'/tag/'selection.json').read_text())
        assert chosen['stop_only'] is True
        for dataset,path in [('confirm','followup/confirm.jsonl'),('regression','data/regression.jsonl')]:
            episodes={e['id']:e for e in map(json.loads,(root/path).read_text().splitlines())}
            by_mode={}
            for mode in ('red_only','auto'):
                f=root/'followup'/tag/f'{dataset}-{mode}.json'
                payload=json.loads(f.read_text());rows=payload['records'];stats=Counter();pairs=defaultdict(list)
                assert {r['id'] for r in rows}==set(episodes)
                assert len(rows)==len(episodes)
                for row in rows:
                    e=episodes[row['id']];code=bytearray.fromhex(e['binary']);stopped=False;proposed=0;rounds=0
                    blue=False
                    for i,action in enumerate(row['actions']):
                        assert action['step']==i
                        assert action['rounds']==(4 if action['blue'] else 64)
                        rounds+=action['rounds'];blue|=action['blue']
                        if action['blue']:
                            assert action['stop'], 'STOP-only policy performed an EDIT'
                        if action['stop']:
                            stopped=True;assert i==len(row['actions'])-1;break
                        proposed+=1
                        if i==8:continue
                        p=action['position']
                        if not 0<=p<len(code):continue
                        assert action['before']==code[p]
                        code[p]=action['value']
                    assert code.hex()==row['final']
                    assert rounds==row['round_examples'] and blue==row['used_blue']
                    numeric=True
                    for inputs,expected in e['cases']:
                        calls+=1
                        try:value=execute_b(bytes(code),inputs)
                        except (ValueError,IndexError):value=None
                        numeric &= value==expected
                    need=bool(e['edits']);strict=bool(code.hex()==e['target'] and stopped and numeric and (need or proposed==0))
                    false=not need and proposed>0
                    assert strict==row['strict'] and numeric==row['numeric'] and false==row['false_edit']
                    assert proposed==row['proposed'] and stopped==row['explicit_stop']
                    stats['total']+=1;stats['edits']+=need;stats['noops']+=not need
                    stats['strict_edits']+=need and strict;stats['strict_total']+=strict;stats['false_edits']+=false
                    stats['blue_episodes']+=blue;stats['round_examples']+=rounds
                    if need and e.get('lane') is not None:pairs[e['pair_id']].append(strict)
                    sequences+=1
                valid=[v for v in pairs.values() if len(v)==2]
                stats['pairs']=len(valid);stats['pairs_both']=sum(all(v) for v in valid)
                for k,v in stats.items():assert v==payload['summary']['all'][k],(f.name,k,v)
                by_mode[mode]={r['id']:r for r in rows}
                files.append({'path':str(f.relative_to(root)),'sha256':file_sha(f),'records':len(rows)})
            b=by_mode['red_only'];a=by_mode['auto']
            calculated={
                'lost_success_ids':{k for k in b if b[k]['strict'] and not a[k]['strict']},
                'added_success_ids':{k for k in b if not b[k]['strict'] and a[k]['strict']},
                'added_false_edit_ids':{k for k in b if not b[k]['false_edit'] and a[k]['false_edit']}}
            recorded=json.loads((root/'followup'/tag/f'{dataset}-comparison.json').read_text())
            for k,v in calculated.items():assert v==set(recorded[k])
            assert not calculated['lost_success_ids'] and not calculated['added_false_edit_ids']
            comparisons.append({'tag':tag,'dataset':dataset,**{k:len(v) for k,v in calculated.items()}})
    result={'passed':True,'record_sequences':sequences,'second_vm_reexecution_calls':calls,
            'models_policies_protocol_unchanged':True,'blue_edit_actions':0,'files':files,'comparisons':comparisons,
            'scope':'same-session independent replay code and supplied second VM; finite inputs, no formal proof or population risk guarantee'}
    (root/'results/stop-audit.json').write_text(json.dumps(result,indent=2))
    print('STOP_AUDIT_PASS',sequences,calls,flush=True)

if __name__=='__main__':main()
