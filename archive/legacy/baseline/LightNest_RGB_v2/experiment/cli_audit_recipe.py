import hashlib,json,os,stat,subprocess,sys
from pathlib import Path
root=Path('/mnt/data/LightNest_RGB_v2')
out=Path('/mnt/data/rgb_v2_work/cli_validation');out.mkdir(exist_ok=False)
rows=[json.loads(x) for x in (root/'baseline_data/calibration.jsonl').read_text().splitlines()]
e=next(x for x in rows if x['family']=='mixed' and x['edits'])
inp=out/'input.rv32';inp.write_bytes(bytes.fromhex(e['binary']));h=hashlib.sha256(inp.read_bytes()).hexdigest()
checks=[]
for seed in (104729,130363):
 for mode in ('red','risk_full'):
  res=[]
  for backend,mod in [('eager','experiment.predict'),('compiled','experiment.predict_compiled')]:
   dest=out/f'{seed}-{mode}-{backend}'
   cmd=[sys.executable,'-m',mod,'--seed',str(seed),'--mode',mode,'--input',str(inp),'--request',e['command'],'--out',str(dest)]
   p=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=30,check=True)
   t=json.loads((dest/'trace.json').read_text())['records'][0]
   assert stat.S_IMODE((dest/'candidate.bin').stat().st_mode)==0o600
   assert stat.S_IMODE((dest/'trace.json').stat().st_mode)==0o600
   assert hashlib.sha256(inp.read_bytes()).hexdigest()==h
   assert (dest/'candidate.bin').read_bytes()==bytes.fromhex(t['final'])
   res.append(t)
   if seed==104729 and mode=='red' and backend=='eager':
    refused=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=30)
    assert refused.returncode!=0
    (out/'existing_directory_rejected.txt').write_text(refused.stderr)
  def actions(x):return [(a['position'],a['value'],a['stop']) for a in x['actions']]
  assert actions(res[0])==actions(res[1]) and res[0]['final']==res[1]['final'] and res[0]['status']==res[1]['status']
  checks.append({'seed':seed,'mode':mode,'discrete_actions_equal':True,'candidate_bytes_equal':True,'files_0600':True})
(root/'results/cli-audit.json').write_text(json.dumps({'passed':True,'checks':checks,'actual_successful_cli_invocations':8,'additional_existing_directory_refusal':1,'source_unchanged':True,'source_sha256':h,'input_id':e['id'],'host_binary_executions':0,'scope':'one fixed public mixed case; CLI integration not new capability score'},indent=2))
print(json.dumps(checks))
