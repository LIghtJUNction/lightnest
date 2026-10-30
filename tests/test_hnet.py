"""Focused tests; no trained checkpoint or external corpus is required."""
import unittest

import torch

from lightnest.generate import condition_tensor, generation_loss
from lightnest.hnet import HNetELFGenerator, HierarchicalGRU, ema_scan, ratio_loss


class HNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_ema_matches_loop_and_gradients(self):
        v = torch.randn(3, 29, 8, requires_grad=True)
        p = torch.rand(3, 29, requires_grad=True)
        init = torch.randn(3, 8)
        last = init; values = []
        for t in range(v.shape[1]):
            last = p[:, t, None] * v[:, t] + (1 - p[:, t, None]) * last
            values.append(last)
        expected = torch.stack(values, 1)
        actual = ema_scan(v, p, init)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        grads_a = torch.autograd.grad(actual.square().sum(), (v, p), retain_graph=True)
        grads_b = torch.autograd.grad(expected.square().sum(), (v, p))
        for a, b in zip(grads_a, grads_b):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    def test_prefix_invariance(self):
        for mode in ('dense', 'fixed', 'dynamic'):
            m = HierarchicalGRU(32, 16, mode=mode).eval()
            x = torch.randn(3, 31, 32)
            y = x.clone(); y[:, 17:] = torch.randn_like(y[:, 17:]) * 5
            a, _ = m(x); pa = m.last_probabilities.clone()
            b, _ = m(y); pb = m.last_probabilities.clone()
            torch.testing.assert_close(a[:, :17], b[:, :17], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(pa[:, :17], pb[:, :17], atol=2e-6, rtol=2e-5)

    def test_full_sequence_matches_stream(self):
        for mode in ('dense', 'fixed', 'dynamic'):
            m = HNetELFGenerator(16, 64, chunk_mode=mode).eval()
            x = torch.randint(0, 256, (3, 31)); x[:, 0] = 256
            c = condition_tensor([{'exit': 3}, {'exit': 14}, {'stdout': '6162636465666768'}])
            full = m(x, c)
            state = None; outputs = []
            for t in range(x.shape[1]):
                result = m(x[:, t:t + 1], c, hidden=state, position_offset=t)
                outputs.append(result); state = result['hidden']
            for key in ('bits', 'blue_bits', 'stop', 'boundary_probabilities'):
                stream = torch.cat([o[key] for o in outputs], 1)
                torch.testing.assert_close(full[key], stream, rtol=2e-4, atol=2e-5)

    def test_arbitrary_stream_chunks(self):
        m = HierarchicalGRU(32, 16).eval(); x = torch.randn(4, 50, 32)
        full, fs = m(x); state = None; out = []
        for a, b in ((0, 3), (3, 14), (14, 15), (15, 37), (37, 50)):
            part, state = m(x[:, a:b], state); out.append(part)
        torch.testing.assert_close(full, torch.cat(out, 1), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(fs.main, state.main, rtol=2e-4, atol=2e-5)

    def test_no_boundary_does_not_update_main(self):
        m = HierarchicalGRU(16, 8, mode='fixed', target_ratio=4)
        _, s = m(torch.randn(2, 1, 16))
        old = s.main.clone()
        _, s2 = m(torch.randn(2, 2, 16), s)
        torch.testing.assert_close(old, s2.main)
        self.assertEqual(m.last_padded_main_positions, 0)
        self.assertEqual(s2.offset, 3)

    def test_fixed_boundaries(self):
        m = HierarchicalGRU(16, 8, mode='fixed', target_ratio=4)
        m(torch.randn(2, 13, 16))
        self.assertEqual(m.last_boundaries[0].nonzero().flatten().tolist(), [0, 4, 8, 12])

    def test_task_loss_trains_router_without_ratio_loss(self):
        m = HNetELFGenerator(16, 64)
        x = torch.randint(0, 256, (3, 25)); t = x.clone(); t[:, -1] = 256
        o = m(x, condition_tensor([{'exit': i} for i in range(3)]))
        generation_loss(o, t, torch.ones_like(t, dtype=torch.bool)).backward()
        for p in (m.decode.query.weight, m.decode.key.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()), 1e-8)

    def test_ratio_minimum_and_mask(self):
        p = torch.full((2, 8), .25, requires_grad=True)
        b = torch.zeros_like(p, dtype=torch.bool); b[:, ::4] = True
        self.assertAlmostEqual(float(ratio_loss(p, b).detach()), 1., places=6)
        v = torch.ones_like(b); v[:, 4:] = False
        self.assertAlmostEqual(float(ratio_loss(p, b, valid=v).detach()), 1., places=6)
        with self.assertRaises(ValueError): ratio_loss(p, b, 1)
        with self.assertRaises(ValueError): ratio_loss(p, b, valid=~torch.ones_like(b))

    def test_parameter_matched_controls(self):
        sizes = []
        for mode in ('dense', 'fixed', 'dynamic'):
            torch.manual_seed(13)
            m = HNetELFGenerator(16, 64, chunk_mode=mode)
            sizes.append({k: v.clone() for k, v in m.state_dict().items()})
        for key in sizes[0]:
            torch.testing.assert_close(sizes[0][key], sizes[1][key])
            torch.testing.assert_close(sizes[0][key], sizes[2][key])

    def test_bad_state_offset_rejected(self):
        m = HNetELFGenerator(16, 64); x = torch.tensor([[256]])
        c = condition_tensor([{'exit': 1}])
        with self.assertRaises(ValueError): m(x, c, position_offset=1)
        o = m(x, c)
        with self.assertRaises(ValueError): m(x, c, hidden=o['hidden'], position_offset=0)

    def test_generation_api(self):
        m = HNetELFGenerator(16, 64).eval()
        with torch.no_grad(): m.stop.weight.zero_(); m.stop.bias.fill_(10)
        result = m.generate({'exit': 5})
        self.assertEqual(result, {'bytes': b'', 'stop': True, 'steps': 1})
        m.train()
        with self.assertRaises(RuntimeError): m.generate({'exit': 5})

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError): HierarchicalGRU(16, 8, mode='unknown')
        with self.assertRaises(ValueError): HierarchicalGRU(16, 8, target_ratio=1)
        with self.assertRaises(ValueError): HierarchicalGRU(16, 8)(torch.empty(1, 0, 16))


if __name__ == '__main__':
    unittest.main()
