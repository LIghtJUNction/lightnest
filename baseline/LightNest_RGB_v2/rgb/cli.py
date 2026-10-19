"""One bounded request -> one next-edit proposal. Writes no executable files."""
import argparse,hashlib,json
from .network import RGBNetwork,load_rgb
from .runtime import common,configure_torch
from .guard import Edit,candidate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',help='one RGB checkpoint; omitted uses unchanged parent plus new modules')
    p.add_argument('--seed',type=int,choices=(104729,130363),default=130363)
    p.add_argument('--binary',required=True,help='hex bytes, up to 1024 bytes')
    p.add_argument('--request',required=True)
    a=p.parse_args();configure_torch()
    code=bytes.fromhex(a.binary)
    m=load_rgb(a.checkpoint) if a.checkpoint else RGBNetwork(common.load_parent(a.seed)).eval()
    encoded=common.encode([{'command':a.request,'binary':code.hex()}],m.red.config)
    result=m(*encoded)
    pos=int(result['scores'][0].argmax())
    base=hashlib.sha256(code).hexdigest()
    out={'base_sha256':base,'source':'BLUE' if bool(result['used_blue'][0]) else 'RED',
         'deployment_approved':False,'action':'STOP' if pos==m.none_index else 'EDIT',
         'red_step_examples':result['red_step_examples']}
    if pos!=m.none_index:
        value=sum(int(result['bits'][0,pos,k]>=0)<<k for k in range(8))
        edit=Edit(pos,code[pos],value)
        try:
            new=candidate(code,base,[edit]);common.graph(new)
            out.update(edit=edit.__dict__,candidate_hex=new.hex(),validation='structural checks only')
        except ValueError as exc:
            out.update(validation='REJECTED',reason=str(exc))
    else:
        out['validation']='STOP is a model decision, not a correctness certificate'
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__': main()
