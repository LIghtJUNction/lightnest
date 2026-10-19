"""Generate a bounded edit sequence, never execute or overwrite a binary.

Default: preserved red path. --research-policy is an explicit experimental
route choice tied to this exact model hash, not a deployment approval.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from rgb.network import load_rgb,RoutePolicy
from rgb.stop_gate import load_stop_only,StopOnlyPolicy
from rgb.runtime import common,configure_torch
from rgb.guard import candidate,Edit
from lightnest_data.core import file_sha

def predict(model,code,request):
    actions=[];status='budget';current=code
    for step in range(9):
        try:
            enc=common.encode([{'binary':current.hex(),'command':request}],model.red.config)
            result=model(*enc);p=int(result['scores'][0].argmax())
        except (ValueError,FloatingPointError) as exc:
            return current,actions,'rejected:'+str(exc)
        a={'step':step,'source':'blue' if bool(result['used_blue'][0]) else 'red','position':p,
           'predicted_blue_error':float(result['predicted_blue_error'][0]),'stop':p==model.none_index}
        actions.append(a)
        if p==model.none_index:return current,actions,'stopped_unverified'
        if step==8:return current,actions,'budget_exhausted'
        if not 0<=p<len(current):return current,actions,'rejected:pointer_out_of_range'
        v=sum(int(result['bits'][0,p,k]>=0)<<k for k in range(8));a.update(before=current[p],after=v)
        try:
            new=candidate(current,hashlib.sha256(current).hexdigest(),[Edit(p,current[p],v)])
            common.graph(new)
        except ValueError as exc:return current,actions,'rejected:'+str(exc)
        current=new
    return current,actions,status

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--request',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--hex');g.add_argument('--input',type=Path)
    p.add_argument('--research-policy',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();configure_torch()
    code=bytes.fromhex(a.hex) if a.hex is not None else a.input.read_bytes()
    if not 4<=len(code)<=1024 or len(code)%4:raise ValueError('restricted RV32 input must be4..1024 aligned bytes')
    m=load_rgb(a.model)
    if a.research_policy:
        s=json.loads(a.research_policy.read_text())
        if s['model_sha256']!=file_sha(a.model):raise ValueError('policy does not match this checkpoint')
        if s.get('stop_only') is True:
            m=load_stop_only(a.model);m.policy=StopOnlyPolicy(s['enabled'],s['threshold'])
        else:
            m.policy=RoutePolicy(s['enabled'],s['threshold'])
    with torch.inference_mode():new,acts,status=predict(m,code,a.request)
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'candidate.bin').open('xb') as f:f.write(new)
    (a.out/'candidate.bin').chmod(0o600)
    trace={'model_sha256':file_sha(a.model),'input_sha256':hashlib.sha256(code).hexdigest(),'output_sha256':hashlib.sha256(new).hexdigest(),
           'status':status,'actions':acts,'route_enabled':m.policy.enabled,'stop_only':getattr(m.policy,'stop_only',False),'native_execution':False,'semantic_validation':False,'deployment_approved':False}
    (a.out/'trace.json').write_text(json.dumps(trace,indent=2));(a.out/'trace.json').chmod(0o600)
    print(json.dumps(trace,indent=2))
if __name__=='__main__':main()
