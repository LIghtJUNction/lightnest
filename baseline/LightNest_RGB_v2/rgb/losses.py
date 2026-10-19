"""Training-only labels: never pass rows, targets, or execution scores to forward.

Router labels here are NEXT-ACTION errors. Full-sequence correctness, no-op
retention, and calibration on separate groups remain required before enabling
fast inference. Red agrees is not the same as red is correct.
"""
import torch
from torch.nn import functional as F
from .runtime import common


def action_correct(scores,bits,rows,none_index):
    positions=scores.detach().argmax(-1).tolist()
    correct=[]
    for j,(pos,row) in enumerate(zip(positions,rows)):
        if pos==none_index:
            correct.append(not row['edits'])
        else:
            value=sum(int(bits[j,pos,k].detach()>=0)<<k for k in range(8))
            correct.append(any(e['offset']==pos and e['value']==value for e in row['edits']))
    return torch.tensor(correct,dtype=torch.bool,device=scores.device)


def rgb_training_loss(outputs,rows,none_index):
    scores,bits=outputs['blue'];red_scores,red_bits=outputs['red']
    supervised,_,_=common.supervised_loss(scores,bits,rows,none_index)
    teacher_ok=action_correct(red_scores,red_bits,rows,none_index)
    if bool(teacher_ok.any()):
        teacher=red_scores[teacher_ok].detach().softmax(-1)
        distill=F.kl_div(scores[teacher_ok].log_softmax(-1),teacher,reduction='batchmean')
    else:
        distill=scores.sum()*0
    wrong=(~action_correct(scores,bits,rows,none_index)).to(scores.dtype)
    route=F.binary_cross_entropy_with_logits(outputs['risk_logits'],wrong)
    total=supervised+0.1*distill+0.2*route
    return total,{'supervised':float(supervised.detach()),'verified_teacher_kl':float(distill.detach()),
                  'router_bce':float(route.detach()),'teacher_correct':int(teacher_ok.sum()),
                  'blue_next_correct':int((wrong==0).sum()),'examples':len(rows)}
