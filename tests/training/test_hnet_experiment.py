"""Check the pilot's evaluation mechanics independently of trained outcomes."""
import unittest

import torch

from lightnest.architecture.generate import ELFGenerator
from lightnest.architecture.hnet import HNetELFGenerator
from lightnest.training.hnet_experiment import batch, generate_batch, teacher_metrics, variable_positions


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_teacher_inputs_are_shifted_and_masked(self):
        rows = [{'raw': b'\x01\x02', 'spec': {'exit': 2}}, {'raw': b'\xff', 'spec': {'exit': 3}}]
        x, _, y, valid = batch(rows)
        self.assertEqual(x.tolist(), [[256, 1, 2], [256, 255, 0]])
        self.assertEqual(y.tolist(), [[1, 2, 256], [255, 256, 0]])
        self.assertEqual(valid.tolist(), [[True, True, True], [True, True, False]])

    def test_batched_generation_matches_existing_single_api(self):
        rows = [{'spec': {'exit': 7}}, {'spec': {'exit': 23}}]
        for cls in (ELFGenerator, HNetELFGenerator):
            torch.manual_seed(51)
            model = cls(width=16, capacity=24).eval()
            with torch.no_grad():
                model.stop.weight.zero_(); model.stop.bias.fill_(-10)
            generated, stopped, _ = generate_batch(model, rows)
            for i, row in enumerate(rows):
                single = model.generate(row['spec'])
                self.assertEqual(generated[i], single['bytes'])
                self.assertEqual(stopped[i], single['stop'])
                self.assertEqual(len(generated[i]), 24)

    def test_empty_immediate_stop(self):
        model = HNetELFGenerator(width=16, capacity=24).eval()
        with torch.no_grad():
            model.stop.weight.zero_(); model.stop.bias.fill_(10)
        generated, stopped, _ = generate_batch(model, [{'spec': {'exit': 1}}])
        self.assertEqual(generated, [b''])
        self.assertEqual(stopped, [True])

    def test_variable_field_detection_ignores_test_data(self):
        rows = []
        for family in ('exit', 'stdout'):
            for split, raw in [('train', b'\x01\x02'), ('train', b'\x01\x03'), ('test', b'\xff\x04')]:
                rows.append({'family': family, 'split': split, 'raw': raw})
        result = variable_positions(rows)
        self.assertEqual(result, {'exit': [1], 'stdout': [1]})

    def test_zero_logits_have_eight_bits_per_byte(self):
        model = ELFGenerator(width=16, capacity=24).eval()
        with torch.no_grad():
            for param in model.parameters(): param.zero_()
        # At zero copy logits, the RGB copy gate is .5, so disable it explicitly.
        with torch.no_grad(): model.copy_gate.bias.fill_(-100)
        rows = [{'raw': b'\x01\x02', 'spec': {'exit': 1}, 'id': 'sample',
                 'family': 'exit', 'sha256': 'unit-test', 'variable_positions': [0]}]
        metrics, _ = teacher_metrics(model, rows)
        self.assertAlmostEqual(metrics['bpb'], 8., places=5)


if __name__ == '__main__':
    unittest.main()
