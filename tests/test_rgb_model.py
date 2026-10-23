from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from lightnest_rgb.artifact import digest
from lightnest_rgb.corpus import compile_generation_target
from lightnest_rgb.execute import Case, ExecutableRunner
from lightnest_rgb.generate import ELFGenerator, condition_tensor, generation_loss
from lightnest_rgb.iterate import Example, RGBEditor
from lightnest_rgb.model import RGBConfig, RGBPolicy, budget_target, patch_loss


class RGBPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(17)
        return RGBPolicy(RGBConfig(width=32, heads=4, landmarks=2, detail_k=2, red_depths=(2, 4)))

    def test_full_artifact_output_alignment_and_gradients(self):
        model = self.model(); original = bytes(range(128)); damaged = bytearray(original); damaged[17] ^= 8
        blue, red, state = model(bytes(damaged), "restore this image", depth=2)
        loss = patch_loss(blue, bytes(damaged), original) + patch_loss(red, bytes(damaged), original)
        loss.backward()
        self.assertEqual(red["logits"].shape, (128,))
        self.assertEqual(red["bits"].shape, (128, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.thought.weight_hh.grad.abs().sum()), 0)
        self.assertGreater(float(model.green.memory.key.weight.grad.abs().sum()), 0)

    def test_fast_acceptance_actually_skips_deeper_compute(self):
        model = self.model().eval()
        iterator = model.thoughts(bytes(range(128)), "repair")
        first, state = next(iterator)
        self.assertEqual(first["origin"], "blue")
        self.assertEqual(model.recurrent_steps, 1)
        self.assertEqual(model.green.memory.bank_builds, 1)
        iterator.close()
        self.assertEqual(model.recurrent_steps, 1)

    def test_red_continues_state_with_one_evidence_bank(self):
        model = self.model().eval()
        outputs = list(model.thoughts(bytes(range(128)), "repair"))
        self.assertEqual([x[0]["depth"] for x in outputs], [1, 2, 4])
        self.assertEqual(model.green.memory.bank_builds, 1)
        self.assertEqual(model.recurrent_steps, 4)

    def test_red_guides_blue_and_emits_language_tokens(self):
        model = self.model().eval()
        state = model.begin(bytes(range(128)), "repair")
        model.advance(state, 2)
        guided = state["guidance"]
        self.assertEqual(guided.shape[-1], model.config.width)
        self.assertEqual(state["abstract"].shape[-1], 4 * model.config.width)
        output = model.readout(state, blue=True)
        self.assertEqual(output["language_logits"].shape, (1, 257))
        self.assertEqual(len(model.blue_language_tokens(state)), 32)

    def test_learned_budget_controls_red_depth_without_input_heuristics(self):
        model = self.model().eval()
        with torch.no_grad():
            model.budget[-1].weight.zero_()
            model.budget[-1].bias.fill_(-10)
            model.budget[-1].bias[2] = 10
        outputs = list(model.thoughts(bytes(range(128)), "repair", budgeted=True))
        self.assertEqual([output["depth"] for output, _ in outputs], [1, 4])
        self.assertEqual(outputs[0][0]["budget_logits"].shape[-1], 3)

    def test_budget_target_uses_task_success(self):
        model = self.model().eval()
        image = bytes(range(128)); target = bytearray(image); target[7] ^= 4
        state = model.begin(image, "repair")
        model.advance(state, 1)
        blue = model.readout(state, blue=True)
        model.advance(state, 2)
        red = model.readout(state, blue=True)
        with torch.no_grad():
            for output in (blue, red):
                output["logits"].fill_(-20)
            red["logits"][7] = 20
            red["bits"][7] = torch.tensor([20 if (target[7] >> bit) & 1 else -20 for bit in range(8)])
        self.assertEqual(budget_target([blue, red], image, bytes(target)), 1)


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_forcing_matches_incremental_decoder(self):
        torch.manual_seed(13); model = ELFGenerator(32, 128).eval()
        tokens = torch.tensor([[256, 127, 69, 76, 70]])
        condition = condition_tensor([{"exit": 37}])
        with torch.no_grad():
            full = model(tokens, condition)
            hidden = None; values = []
            for i in range(tokens.shape[1]):
                out = model(tokens[:, i:i + 1], condition, hidden=hidden, position_offset=i)
                hidden = out["hidden"]; values.append(out["bits"])
        torch.testing.assert_close(full["bits"], torch.cat(values, 1), rtol=1e-4, atol=1e-5)

    def test_generation_training_and_request_boundaries(self):
        model = ELFGenerator(32, 128)
        previous = torch.tensor([[256, 127, 69, 76]])
        targets = torch.tensor([[127, 69, 76, 256]])
        output = model(previous, condition_tensor([{"stdout": b"abcdefgh".hex()}]))
        loss = generation_loss(output, targets, torch.ones_like(targets, dtype=torch.bool)); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.red.weight_hh.grad.abs().sum()), 0)
        for spec in ({"exit": -1}, {"exit": True}, {"stdout": "00"}, {"exit": 1, "binary": "7f454c46"}):
            with self.assertRaises(ValueError): condition_tensor([spec])

    def test_learned_stop_produces_no_automatic_header(self):
        model = ELFGenerator(32, 128).eval()
        with torch.no_grad(): model.stop.weight.zero_(); model.stop.bias.fill_(20)
        output = model.generate({"exit": 3})
        self.assertTrue(output["stop"]); self.assertEqual(output["bytes"], b"")


class IterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        original = compile_generation_target({"exit": 37})
        damaged = bytearray(original); damaged[0] ^= 1; damaged[18] ^= 1
        class Policy:
            training = False
            recurrent_steps = 0
            green = SimpleNamespace(encoded_chunks=0, memory=SimpleNamespace(bank_builds=0))
            def __init__(self): self.inputs = []
            def thoughts(self, image, command, frames, **kwargs):
                self.inputs.append(digest(image)); self.green.memory.bank_builds += 1
                offset = 0 if image[0] != original[0] else 18
                for depth in (1, 2, 4):
                    self.recurrent_steps = self.recurrent_steps + depth - (0 if depth == 1 else depth // 2)
                    logits = torch.full((len(image),), -20.); logits[offset] = 20.
                    bits = torch.zeros(len(image), 8)
                    bits[offset] = torch.tensor([20. if original[offset] >> k & 1 else -20. for k in range(8)])
                    yield {"logits": logits, "bits": bits, "risk_logit": torch.tensor(-20.), "depth": depth,
                           "origin": "blue" if depth == 1 else "red"}, {"observation": SimpleNamespace(instances=3, payloads=[b"a", b"b"])}
        return original, bytes(damaged), Policy()

    def test_two_steps_reobserve_changed_image_and_keep_source(self):
        original, damaged, policy = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            runner = ExecutableRunner(Path(temp))
            reference = runner.run(original, Case())
            expected = {k: reference[k] for k in ("status", "returncode", "stdout", "stderr")}
            result = RGBEditor(policy, runner, max_candidates=24).repair(damaged, "repair", [Example(Case(), expected)])
        self.assertTrue(result["stop"]); self.assertEqual(result["bytes"], original)
        self.assertEqual(result["edits"], 2); self.assertEqual(result["bank_builds"], 2)
        self.assertEqual(len(set(policy.inputs)), 2)
        patches = [x for x in result["trace"] if x["action"] == "PATCH"]
        self.assertEqual(patches[0]["after"], patches[1]["before"])
        self.assertEqual(digest(damaged), result["source_sha256"])

    def test_visible_precheck_is_not_called_neural_fast_acceptance(self):
        original, _, policy = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            runner = ExecutableRunner(Path(temp)); r = runner.run(original, Case())
            result = RGBEditor(policy, runner).repair(original, "repair", [Example(Case(), {k:r[k] for k in ("status", "returncode", "stdout", "stderr")})])
        self.assertEqual(result["recurrent_steps"], 0)
        self.assertEqual(result["trace"][0]["origin"], "visible_precheck")


if __name__ == "__main__":
    unittest.main()
