import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.active.elf_generation import run
import torch
from experiments.active.elf_generation.run import tensors, variable_fields, measure, rank
from experiments.active.elf_generation.models import Candidate


class ProtocolTests(unittest.TestCase):
    def test_state_replaces_previous_state_without_history(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run, 'ROOT', Path(directory)):
                run.state(phase='training', round='first', incumbent={'variant':'mlp'})
                run.state(phase='round_complete', round='second', incumbent=None)
                state=json.loads((Path(directory)/'experiments/STATUS.json').read_text())
                self.assertEqual(state['round'],'second')
                self.assertEqual(state['incumbent'],{'variant':'mlp'})
                self.assertNotIn('history',state)
                self.assertNotIn('first',json.dumps(state))

    def test_every_declared_variant_is_registered(self):
        plan=json.loads((run.ROOT/'experiments/PLAN.json').read_text())
        for item in plan['rounds']:
            arguments=item['args']
            if '--variants' in arguments:
                for variant in arguments[arguments.index('--variants')+1:]:
                    if variant.startswith('--'): break
                    self.assertIn(variant,run.CANDIDATE_NAMES)

    def test_expanded_validation_keeps_splits_disjoint(self):
        with patch.object(run, 'compile_target', lambda spec: json.dumps(spec,sort_keys=True).encode()):
            rows=run.dataset(601,stdout_validation=128,stdout_test=128)
        counts={split:sum(row['split']==split for row in rows) for split in ('train','validation','test')}
        self.assertEqual(counts,{'train':384,'validation':160,'test':160})
        self.assertEqual(len({row['sha256'] for row in rows}),len(rows))

    def test_variable_fields_use_only_training(self):
        rows = [dict(family=f, split=s, raw=raw) for f in ('exit','stdout')
                for s,raw in [('train',b'\x00\x01'),('train',b'\x00\x02'),('test',b'\xff\x03')]]
        self.assertEqual(variable_fields(rows), {'exit':[1],'stdout':[1]})

    def test_full_file_requires_first_stop_at_correct_length(self):
        rows = [dict(id=f, family=f, raw=b'\x03', spec={'exit':3}, sha256='reference') for f in ('exit','stdout')]
        packed=tensors(rows,2)
        class Oracle(torch.nn.Module):
            capacity=2
            def forward(self, condition):
                output=torch.full((len(condition),3,9),-20.)
                output[:,0,:2]=20.; output[:,1,8]=20.
                return output
        score,_=measure(Oracle(),rows,packed,{'exit':[0],'stdout':[0]},torch.ones(3))
        self.assertEqual(score['macro_exact'],1.)
        class NeverStop(Oracle):
            def forward(self, condition):
                output=super().forward(condition); output[:,:,8]=-20.; return output
        score,_=measure(NeverStop(),rows,packed,{'exit':[0],'stdout':[0]},torch.ones(3))
        self.assertEqual(score['exact'],0)

    def test_mode_only_stop_is_invariant_to_payload(self):
        torch.manual_seed(11)
        for variant in ('conv_copy_mode_small','mlp_copy_mode_small'):
            model=Candidate(variant,32,16).eval()
            condition=torch.zeros(2,10,dtype=torch.long);condition[:,0]=1
            condition[1,2:]=255
            with torch.inference_mode(): output=model(condition)
            torch.testing.assert_close(output[0,:,8],output[1,:,8],atol=0,rtol=0)

    def test_mode_only_copy_is_affine_in_requested_bits(self):
        model=Candidate('mlp_copy_mode_small',32,16).eval()
        condition=torch.zeros(4,10,dtype=torch.long);condition[:,0]=1
        condition[1,2]=255;condition[2,3]=255;condition[3,2:4]=255
        with torch.inference_mode(): output=model(condition)
        torch.testing.assert_close(output[0]+output[3],output[1]+output[2],atol=2e-6,rtol=1e-5)

    def test_primary_metric_dominates_low_loss(self):
        success=dict(macro_exact=.1,variable_accuracy=0.,loss=10.)
        failure=dict(macro_exact=0.,variable_accuracy=1.,loss=.001)
        self.assertLess(rank(success),rank(failure))

    def test_all_candidates_backpropagate_without_target_inputs(self):
        previous=torch.get_num_threads(); torch.set_num_threads(1)
        try:
            for variant in ('mlp','gru','conv','transformer','mlp_copy','conv_copy','mlp_copy_small','mlp_copy_bias_small','mlp_copy_bias_tiny','conv_copy_mode_small','mlp_copy_mode_small'):
                model=Candidate(variant,8,16)
                output=model(torch.zeros(2,10,dtype=torch.long))
                self.assertEqual(output.shape,(2,17,9))
                output.square().mean().backward()
                self.assertIsNotNone(model.request.weight.grad)
        finally:
            torch.set_num_threads(previous)
