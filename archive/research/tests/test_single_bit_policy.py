"""Single-bit actions must obey the task contract and retain learned gradients."""
import unittest

import torch

from lightnest.architecture.model import RGBConfig, RGBPolicy
from archive.research.single_bit_policy import SingleBitRGBPolicy
from archive.research.tools.research_exact_repair import categorical_loss


class SingleBitPolicyTests(unittest.TestCase):
    def test_every_proposed_byte_flips_exactly_one_learned_bit(self):
        config = RGBConfig(width=32, landmarks=2, detail_k=2, red_depths=(2,))
        model = SingleBitRGBPolicy(config).eval(); image = bytes(range(128))
        with torch.no_grad():
            for head in (model.blue, model.red):
                head.value.weight.zero_(); head.value.bias.fill_(-20); head.value.bias[6] = 10
        for output, state in model.thoughts(image, 'restore the single-bit fault'):
            outputs = [output] + ([state['guided_blue']] if 'guided_blue' in state else [])
            for prediction in outputs:
                actual = bytes(sum(int(value >= 0) << bit for bit, value in enumerate(row)) for row in prediction['bits'])
                self.assertEqual(actual, bytes(value ^ 64 for value in image))
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in RGBPolicy(config).parameters()))

    def test_categorical_loss_differentiates_location_and_flip_choice(self):
        image, target = bytes(4), b'\0\0\x20\0'
        output = {'logits': torch.zeros(4, requires_grad=True), 'flip_logits': torch.zeros(4, 8, requires_grad=True)}
        loss = categorical_loss(output, image, target); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(output['logits'].grad[2]), 0)
        self.assertLess(float(output['flip_logits'].grad[2, 5]), 0)
        self.assertGreater(float(output['flip_logits'].grad[2, 0]), 0)
        with self.assertRaisesRegex(ValueError, 'one changed bit'):
            categorical_loss(output, image, b'\0\0\x21\0')

    def test_complete_attention_still_receives_full_artifact_and_feedback(self):
        model = SingleBitRGBPolicy(RGBConfig(width=32, landmarks=2, detail_k=2, red_depths=(2,)))
        blue, _, _ = model(bytes(range(128)), 'restore the single-bit fault', mode='dense', depth=2)
        self.assertEqual(blue['logits'].shape, (128,))
        categorical_loss(blue, bytes(range(128)), bytes([0 ^ 2, *range(1, 128)])).backward()
        self.assertTrue(torch.isfinite(model.green.memory.query.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
