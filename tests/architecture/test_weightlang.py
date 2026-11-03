"""Integration contracts for WGT-backed public model entrypoints."""
import tempfile
import unittest
from pathlib import Path

import torch
from lightnest.architecture.generate import ELFGenerator, condition_tensor
from lightnest.architecture.model import RGBConfig, RGBPolicy
from lightnest.architecture.weightlang_io import save_model, load_model
from lightnest.training.training import load_generator, load_policy, math_sources
from lightnest.self_hosting.unified import RGBArtifactAgent


class WeightLangTests(unittest.TestCase):
    def test_public_generator_roundtrip_and_source_authority(self):
        from weightlang.tensor_program import parse_tensor_program
        import lightnest.architecture.generate as host
        model=ELFGenerator(width=8,capacity=16).eval()
        x=torch.tensor([[256,7,8]])
        condition=condition_tensor([{'exit':7}])
        expected=model(x,condition)
        source=type(model)._wgt_source
        changed=source.replace('"bits": logits,', '"bits": logits * 0,')
        program=parse_tensor_program(changed)
        changed_cls=program.classes({name:getattr(host,name) for name in program.exports})['ELFGenerator']
        changed_model=changed_cls(8,16).eval(); changed_model.load_state_dict(model.state_dict())
        self.assertEqual(torch.count_nonzero(changed_model(x,condition)['bits']),0)
        self.assertGreater(torch.count_nonzero(expected['bits']),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'generator'
            save_model(model,path,metadata={'test':'roundtrip'},origin='random-init')
            reloaded,meta=load_generator(path)
            self.assertEqual(meta,{'test':'roundtrip'})
            torch.testing.assert_close(reloaded(x,condition)['bits'],expected['bits'])
            agent=RGBArtifactAgent(generator_checkpoint=path,workdir=Path(tmp)/'work')
            torch.testing.assert_close(agent.generator(x,condition)['bits'],expected['bits'])

    def test_config_state_and_backward_survive_public_policy_io(self):
        config=RGBConfig(width=16,chunk_size=8,heads=2,landmarks=2,detail_k=2,note_slots=2,red_depths=(2,4))
        model=RGBPolicy(config).eval()
        before=model(b'abcdefghijk','edit')[1]['bits']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy'; save_model(model,path,origin='random-init')
            loaded,_=load_policy(path)
            self.assertEqual(loaded.config,config)
            torch.testing.assert_close(loaded(b'abcdefghijk','edit')[1]['bits'],before)
            loaded.train(); loss=loaded(b'abcdefghijk','edit')[1]['bits'].square().mean()
            loss.backward()
            self.assertIsNotNone(loaded.thought.weight_hh.grad)
            self.assertTrue(any(name.endswith('.wgt') for name in math_sources('policy')))


if __name__=='__main__': unittest.main()
