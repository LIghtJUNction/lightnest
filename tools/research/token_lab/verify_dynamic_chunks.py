"""Reload final checkpoints and instrument actual conditional slow-state execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dynamic_chunks import ARMS, Memory, example, evaluate

p=argparse.ArgumentParser()
p.add_argument('directory',nargs='?',default='results/dynamic-chunks')
a=p.parse_args();root=Path(a.directory)
config=json.loads((root/'config.json').read_text())
torch.set_num_threads(1)
examples=[example(np.random.default_rng(80000+j),4) for j in range(config['eval_size'])]
checks=[]
for arm in ARMS:
    for seed in config['seeds']:
        model=Memory(arm,config['hidden'],config['period'])
        model.load_state_dict(torch.load(root/f'{arm}-seed{seed}.pt',weights_only=True))
        executed=[0]
        def hook(module,args):
            executed[0]+=args[0].shape[0]
        handle=model.slow.register_forward_pre_hook(hook)
        observed=evaluate(model,examples)
        handle.remove()
        saved=json.loads((root/f'{arm}-seed{seed}.json').read_text())['metrics']['id']
        for key in ['answer_accuracy','exact_match','slow_update_fraction']:
            assert abs(observed[key]-saved[key])<1e-12,(arm,seed,key)
        assert executed[0]==observed['slow_cell_rows_executed']
        demo_max=sum(sum(1 for length in np.diff([-1]+[i for i,b in enumerate(d['boundaries']) if b]) if length==32) for d in observed['demos'])
        checks.append({'arm':arm,'seed':seed,'accuracy_reproduced':True,'actual_slow_rows':executed[0],
                       'max_length_chunks_in_first_three_demos':demo_max})
        print('Verified checkpoint and actual slow-cell execution:',arm,seed,flush=True)
(root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
