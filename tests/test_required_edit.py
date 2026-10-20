import importlib.util
import unittest
from pathlib import Path
import torch

PATH = Path(__file__).resolve().parents[1] / 'experiments/required_edit_v1/loss.py'
spec = importlib.util.spec_from_file_location('required_edit_loss', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loss = module.required_edit_loss


class RequiredEditTests(unittest.TestCase):
    def inputs(self):
        scores = torch.tensor([[8., -8., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        bits = torch.zeros(1, 4, 8, dtype=torch.float64, requires_grad=True)
        rows = [{'edits': [{'offset': 0, 'value': 0}, {'offset': 1, 'value': 64}]}]
        return scores, bits, rows

    def test_baseline_formula(self):
        z, b, r = self.inputs()
        actual = loss(z, b, r, 4)[1]
        expected = -(z.log_softmax(-1)[0, :2]).logsumexp(0)
        self.assertTrue(torch.equal(actual, expected))

    def test_single_edit_identical(self):
        z, b, r = self.inputs(); r[0]['edits'] = r[0]['edits'][:1]
        vals = [loss(z, b, r, 4, v)[0] for v in module.VARIANTS]
        self.assertTrue(all(torch.equal(vals[0], v) for v in vals))

    def test_stop_identical(self):
        z, b, r = self.inputs(); r[0]['edits'] = []
        vals = [loss(z, b, r, 4, v)[0] for v in module.VARIANTS]
        self.assertTrue(all(torch.equal(vals[0], v) for v in vals))

    def test_set_mass_gradient_is_self_weighted(self):
        z, b, r = self.inputs(); p = z.softmax(-1).detach()[0]
        grad, = torch.autograd.grad(loss(z, b, r, 4)[1], z)
        expected = p.clone(); expected[:2] -= p[:2] / p[:2].sum()
        torch.testing.assert_close(grad[0], expected)
        self.assertLess(abs(float(grad[0, 1])), abs(float(grad[0, 0])) * 1e-6)

    def test_coverage_gradient_supports_weak_required_edit(self):
        z, b, r = self.inputs(); p = z.softmax(-1).detach()[0]
        grad, = torch.autograd.grad(loss(z, b, r, 4, 'coverage')[1], z)
        expected = p.clone(); expected[:2] -= .5
        torch.testing.assert_close(grad[0], expected)
        self.assertLess(float(grad[0, 1]), -.49)

    def test_coverage_decomposition(self):
        z, b, r = self.inputs()
        mass = loss(z, b, r, 4, 'set_mass')[1]
        cover = loss(z, b, r, 4, 'coverage')[1]
        conditional_logp = z[0, :2].log_softmax(0)
        torch.testing.assert_close(cover, mass - conditional_logp.mean())

    def test_required_order_does_not_matter(self):
        z, b, r = self.inputs()
        for v in module.VARIANTS:
            a = loss(z, b, r, 4, v)[0]
            c = loss(z, b, [{'edits': list(reversed(r[0]['edits']))}], 4, v)[0]
            self.assertTrue(torch.equal(a, c))

    def test_byte_loss_unchanged(self):
        z, b, r = self.inputs()
        vals = [loss(z, b, r, 4, v)[2] for v in module.VARIANTS]
        self.assertTrue(all(torch.equal(vals[0], v) for v in vals))

    def test_invalid_labels_rejected(self):
        z, b, _ = self.inputs()
        for edits in ([{'offset': 4, 'value': 0}], [{'offset': -1, 'value': 0}],
                      [{'offset': 0, 'value': 256}], [{'offset': True, 'value': 0}],
                      [{'offset': 0, 'value': 0}] * 2):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                loss(z, b, [{'edits': edits}], 4)

    def test_unknown_rule_rejected(self):
        z, b, r = self.inputs()
        with self.assertRaises(ValueError): loss(z, b, r, 4, 'guess')

    def test_worst_required_gets_feedback(self):
        z, b, r = self.inputs()
        grad, = torch.autograd.grad(loss(z, b, r, 4, 'weakest')[1], z)
        self.assertLess(float(grad[0, 1]), -.99)

    def test_invalid_batch_rejected(self):
        z, b, r = self.inputs()
        with self.assertRaises(ValueError): loss(z, b, [], 4)


if __name__ == '__main__': unittest.main()
