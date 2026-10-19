"""Separate readonly scoring reconstruction, no import of study scoring code.

This is another script in the same session, not third-party certification.
It replays saved discrete edits and executes the second bounded decoder.
"""
import json,hashlib
from collections import defaultdict,Counter
from . import ROOT,BASE
from .prepare import read,write,sha
from .model import decode_action,encode_action
from lightnest_data.vm import execute_b

def main():
    f=json.loads((ROOT/'final-freeze.json').read_text())
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    counts={'sequences':0,'second_vm_calls':0,'text_roundtrips':0};audits=[]
    for p in sorted((ROOT/'results').glob('final-*.json')):
        d=json.loads(p.read_text());es=read(ROOT/'data/test.jsonl' if d['split']=='test' else BASE/'baseline_data/regression.jsonl');em={e['id']:e for e in es}
        assert len(em)==len(d['records'])==len(es)
        recomputed=[]
        for r in d['records']:
            e=em[r['id']];code=bytearray.fromhex(e['binary']);stopped=False;edits=0;rounds=0
            for i,a in enumerate(r['actions']):
                assert not stopped;rounds+=a['rounds'];assert 1<=a['rounds']<=64
                if 'text' in a:
                    texta=decode_action(a['text']);assert all(texta[k]==a[k] for k in ('position','value','stop'));counts['text_roundtrips']+=1
                if a['stop']:stopped=True;break
                edits+=1
                if i>=8:break
                pos=a['position'];v=a['value']
                if not 0<=pos<len(code):break
                assert type(v) is int and 0<=v<256
                if v==code[pos]:break
                assert a.get('before',code[pos])==code[pos];code[pos]=v
            assert code.hex()==r['final'];assert stopped==r['explicit_stop'];assert rounds==r['round_examples']
            numeric=True
            for x,y in e['cases']:
                try:got=execute_b(bytes(code),x)
                except (ValueError,IndexError):got=None
                counts['second_vm_calls']+=1
                if got!=y:numeric=False
            strict=code.hex()==e['target'] and stopped and numeric and (bool(e['edits']) or edits==0)
            false=not e['edits'] and edits>0
            assert strict==r['strict'] and false==r['false_edit'] and numeric==r['numeric'],(p.name,r['id'])
            recomputed.append((r['id'],strict,false));counts['sequences']+=1
        all_=d['summary']['all'];assert sum(s for _,s,_ in recomputed)==all_['strict_total'];assert sum(b for _,_,b in recomputed)==all_['false_edits']
        audits.append({'file':p.name,'rows':len(recomputed),'passed':True})
    for seed in (104729,130363):
        s=json.loads((ROOT/f'results/sft-{seed}.json').read_text());assert s['changed']['red']==s['changed']['green']==0
        for mode in ('rl0','rlcost'):
            r=json.loads((ROOT/f'results/rl-{mode}-{seed}.json').read_text());assert len(r['steps'])==96
            assert r['episodes']==96*4*4
            assert r['max_replay_logp_error']<1e-4
            assert r['changed_policy_values']>0
    write(ROOT/'results/audit.json',{'passed':True,**counts,'files':audits,'frozen_models_and_sources_unchanged':True,'same_session_separate_script':True})
    print('AUDIT',counts,flush=True)
if __name__=='__main__':main()
