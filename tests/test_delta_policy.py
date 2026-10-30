"""The learned XOR head must copy and flip raw input bytes without a target."""
import unittest

import torch

from archive.research.delta_policy import DeltaRGBPolicy
from lightnest.model import RGBConfig, RGBPolicy


class DeltaPolicyTests(unittest.TestCase):
    def model(self, flip=None):
        model = DeltaRGBPolicy(RGBConfig(width=32, heads=4, landmarks=2, detail_k=2, red_depths=(2,))).eval()
        with torch.no_grad():
            for head in (model.blue, model.red):
                head.value.weight.zero_(); head.value.bias.fill_(-20)
                if flip is not None: head.value.bias[flip] = 20
        return model

    def predictions(self, output):
        return bytes(sum(int(value >= 0) << bit for bit, value in enumerate(row)) for row in output['bits'])

    def test_no_flip_copies_arbitrary_input_bytes(self):
        image = bytes(range(128)); model = self.model()
        with torch.inference_mode():
            blue, red, _ = model(image, 'restore behavior', depth=2)
        self.assertEqual(self.predictions(blue), image)
        self.assertEqual(self.predictions(red), image)

    def test_one_learned_bit_flip_is_applied_in_all_thought_paths(self):
        image = bytes(range(128)); model = self.model(flip=2)
        expected = bytes(value ^ 4 for value in image)
        outputs = list(model.thoughts(image, 'restore behavior'))
        for output, state in outputs:
            self.assertEqual(self.predictions(output), expected)
            if 'guided_blue' in state:
                self.assertEqual(self.predictions(state['guided_blue']), expected)

    def test_representation_adds_no_parameters(self):
        config = RGBConfig(width=32, heads=4, landmarks=2, detail_k=2, red_depths=(2,))
        ordinary, delta = RGBPolicy(config), DeltaRGBPolicy(config)
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()), sum(p.numel() for p in delta.parameters()))


if __name__ == '__main__':
    unittest.main()
