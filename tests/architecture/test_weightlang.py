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
        from weightlang import compile_source
        from weightlang.modules import classes_from_ir
        import lightnest.architecture.generate as host
        model=ELFGenerator(width=8,capacity=16).eval()
        x=torch.tensor([[256,7,8]])
        condition=condition_tensor([{'exit':7}])
        expected=model(x,condition)
        source=type(model)._wgt_source
        changed=source.replace("'bits': logits,", "'bits': logits * 0,")
        program=compile_source(changed)
        changed_cls=classes_from_ir(program, vars(host), trusted=True)['ELFGenerator']
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

    def test_directory_inference_loaders_select_eval_mode(self):
        from lightnest.architecture.any2any import ArtifactCore
        from lightnest.architecture.weightlang_io import load_any2any
        config = RGBConfig(width=16, chunk_size=8, heads=2, landmarks=2, detail_k=2,
                           note_slots=2, red_depths=(2, 4))
        cases = [
            ("any2any", ArtifactCore(width=32, heads=4, sequence_limit=256, note_slots=2), load_any2any),
            ("policy", RGBPolicy(config), load_policy),
            ("generator", ELFGenerator(width=8, capacity=16), load_generator),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, model, loader in cases:
                with self.subTest(model=name):
                    path = Path(tmp) / name
                    save_model(model.train(), path, origin="random-init")
                    loaded, _ = loader(path)
                    self.assertFalse(loaded.training)
                    self.assertTrue(all(not module.training for module in loaded.modules()))

    def test_previous_wgt_checkpoint_source_pins_remain_loadable(self):
        import json
        from lightnest.training.training import ROOT, save_generator
        pins = json.loads((ROOT / "lightnest/architecture/wgt/legacy-source-hashes.json").read_text())["sha256"]
        recorded = {name: digest for name, digest in pins.items()
                    if name.endswith(".wgt") or name == "lightnest/architecture/generate.py"}
        model = ELFGenerator(width=8, capacity=16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generator.pt"
            save_generator(path, model, {"math_sources": recorded})
            loaded, _ = load_generator(path)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)
            recorded["lightnest/architecture/generate.py"] = "unknown-source"
            rejected = Path(tmp) / "changed.pt"
            save_generator(rejected, model, {"math_sources": recorded})
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_generator(rejected)


if __name__=='__main__': unittest.main()
