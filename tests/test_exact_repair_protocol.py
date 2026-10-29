"""A failed public-validation gate must never instantiate a test policy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lightnest_rgb.artifact import digest
from tools.research_exact_repair import ROOT, evaluate, hashes, ranking_loss, refreeze, verify


class RepairProtocolTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'tasks.jsonl').write_text('')
        (root / 'counterexamples.json').write_text('[]')
        protocol = {'sources': hashes(), 'tasks_sha256': digest(b''),
                    'counterexamples_sha256': digest(b'[]')}
        (root / 'protocol.json').write_text(json.dumps(protocol))
        for name in hashes():
            destination = root / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        (root / 'training').mkdir()
        torch.save({'sources': hashes(), 'protocol_sha256': digest((root / 'protocol.json').read_bytes()),
                    'validation': {'rate': 0.0}, 'initial_rate': 0.0}, root / 'training/candidate.pt')

    def test_failed_validation_does_not_open_neural_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixture(root)
            with patch('tools.research_exact_repair.RGBPolicy', side_effect=AssertionError('test policy opened')):
                result = evaluate(root)
            self.assertEqual(result['test_queries'], 0)
            self.assertEqual(result['reason'], 'validation_gate_failed')
            self.assertFalse((root / 'evaluation').exists())

    def test_changed_frozen_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixture(root)
            (root / 'tasks.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'frozen repair experiment changed'):
                verify(root)

    def test_refreeze_preserves_data_and_rejects_used_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); parent = base / 'parent'; parent.mkdir(); self.fixture(parent)
            result = refreeze(base / 'new', parent, steps=256, objective='ranking')
            self.assertEqual(result['tasks_sha256'], digest(b''))
            self.assertEqual(result['test_queries'], 0)
            self.assertEqual((base / 'new/tasks.jsonl').read_bytes(), b'')
            (parent / 'review.json').write_text(json.dumps({'test_queries': 1}))
            with self.assertRaisesRegex(ValueError, 'previously queried tests'):
                refreeze(base / 'blocked', parent, steps=256, objective='ranking')

    def test_ranking_objective_rewards_the_correct_action(self):
        image, target = bytes(3), b'\0\0\1'
        neutral = {'logits': torch.zeros(3, requires_grad=True), 'bits': torch.zeros(3, 8, requires_grad=True)}
        loss = ranking_loss(neutral, image, target); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(neutral['logits'].grad[2]), 0)
        correct = {'logits': torch.tensor([-10., -10., 10.]),
                   'bits': torch.tensor([[0.] * 8, [0.] * 8, [10.] + [-10.] * 7])}
        self.assertLess(float(ranking_loss(correct, image, target)), float(loss.detach()))


if __name__ == '__main__':
    unittest.main()
