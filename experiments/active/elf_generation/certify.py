"""Bound all request-bit combinations for a mode-only, affine copy decoder.

This certifies a train-inferred fixed-layout byte contract, not arbitrary ELF
semantics. Floating-point margin tolerance is explicit; no model is repaired.
"""
import argparse
import json
from pathlib import Path
import torch
from weightlang.modules import load_module
from .tasks import condition_tensor, compile_target
from .run import write

ROOT=Path(__file__).resolve().parents[3]


@torch.inference_mode()
def certify(model, rows, tolerance=.001):
    if not getattr(model,'mode_only',False) or not model.transfer:
        return dict(applicable=False,reason='requires payload-independent backbone and affine request-copy head')
    result={}
    for family,mode,columns in [('exit',0,[1]),('stdout',1,list(range(2,10)))]:
        selected=[r for r in rows if r['family']==family and r['split']=='train']
        references=[compile_target(r['spec']) for r in selected]
        if len({len(raw) for raw in references})!=1: raise ValueError('layout lengths differ within family')
        n=len(references[0]); target=torch.tensor([list(raw) for raw in references])
        request=condition_tensor([r['spec'] for r in selected])
        probes=torch.zeros(len(columns)+1,10,dtype=torch.long);probes[:,0]=mode
        for i,column in enumerate(columns,1): probes[i,column]=255
        output=model(probes);base=output[0,:n,:8]
        coefficients=output[1:,:n,:8]-base[None]
        lower=base+coefficients.clamp(max=0).sum(0)
        upper=base+coefficients.clamp(min=0).sum(0)
        bad=[];margins=[];variable_mapping={}
        for position in range(n):
            values=target[:,position]
            if bool(values.eq(values[0]).all()):
                expected=((values[0]>>torch.arange(8))&1).bool()
                margin=torch.where(expected,lower[position],-upper[position])
            else:
                matches=[i for i,column in enumerate(columns) if torch.equal(values,request[:,column])]
                if len(matches)!=1: raise ValueError(f'cannot infer unique request slot at {position}')
                index=matches[0];variable_mapping[str(position)]=columns[index]
                other=[i for i in range(len(columns)) if i!=index]
                one_min=base[position]+coefficients[index,position]
                zero_max=base[position].clone()
                if other:
                    one_min=one_min+coefficients[other,position].clamp(max=0).sum(0)
                    zero_max=zero_max+coefficients[other,position].clamp(min=0).sum(0)
                margin=torch.minimum(one_min,-zero_max)
            margins.append(float(margin.min()))
            if bool((margin<=tolerance).any()):bad.append(position)
        stop=output[0,:,8]
        stop_margin=min(float(-stop[:n].max()),float(stop[n]))
        invariant=bool(torch.allclose(output[:,:,8],output[:1,:,8].expand_as(output[:,:,8]),atol=1e-6,rtol=0))
        result[family]=dict(passed=not bad and stop_margin>tolerance and invariant,
                            target_bytes=n,minimum_bit_margin=min(margins),stop_margin=stop_margin,
                            uncertain_byte_positions=bad,stop_payload_invariant=invariant,
                            learned_request_mapping=variable_mapping)
    return dict(applicable=True,passed=all(r['passed'] for r in result.values()),tolerance=tolerance,
                scope='all request-bit combinations under the train-inferred two-layout contract; not arbitrary program correctness',
                families=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--round',required=True)
    args=parser.parse_args()
    if not args.round.replace('-','').isalnum():parser.error('invalid round')
    archive=ROOT/'experiments/archive'/args.round
    model=load_module(ROOT/'weights/experiments'/args.round/'model',trusted=True).module.eval()
    torch.set_num_threads(1)
    rows=json.loads((archive/'questions.json').read_text())
    result=certify(model,rows)
    write(archive/'copy-margin-certificate.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
