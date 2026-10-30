import unittest

import torch

from lightnest.attention import CompressRouteAttention, joint_attention
from lightnest.observe import Frame, GreenObserver, ObservationStore


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_and_detail_use_one_normalization(self):
        torch.manual_seed(7)
        query = torch.randn(2, 3, 4, 8, dtype=torch.float64)
        shared_k, shared_v = (torch.randn(2, 3, 5, 8, dtype=torch.float64) for _ in range(2))
        detail_k, detail_v = (torch.randn(2, 3, 4, 6, 8, dtype=torch.float64) for _ in range(2))
        valid = torch.rand(2, 3, 4, 6) > .3
        actual = joint_attention(query, shared_k, shared_v, detail_k, detail_v, valid)
        keys = torch.cat((shared_k[:, :, None].expand(-1, -1, 4, -1, -1), detail_k), -2)
        values = torch.cat((shared_v[:, :, None].expand(-1, -1, 4, -1, -1), detail_v), -2)
        mask = torch.cat((torch.ones(2, 3, 4, 5, dtype=torch.bool), valid), -1)
        scores = (query[..., None, :] * keys).sum(-1) / 8**.5
        expected = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None] * values).sum(-2)
        torch.testing.assert_close(actual, expected)

    def test_padding_and_differentiable_shared_context(self):
        torch.manual_seed(11)
        module = CompressRouteAttention(32, 4, 3, 4)
        x = torch.randn(2, 9, 32, requires_grad=True)
        valid = torch.tensor([[True] * 9, [True] * 2 + [False] * 7])
        condition = torch.randn(2, 32, requires_grad=True)
        bank = module.build(x, valid, condition)
        query = torch.randn(2, 5, 32, requires_grad=True)
        result = module.read(query, bank)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.equal(x.grad[1, 2:], torch.zeros_like(x.grad[1, 2:])))
        self.assertGreater(float(condition.grad.abs().sum()), 0)
        self.assertEqual(module.bank_builds, 1)
        self.assertEqual(bank.detail_indices.shape, (2, 4, 3, 4))

    def test_thinking_reuses_bank_but_recomputes_query(self):
        module = CompressRouteAttention(32, 4, 3, 4)
        x = torch.randn(1, 13, 32); valid = torch.ones(1, 13, dtype=torch.bool)
        bank = module.build(x, valid, torch.zeros(1, 32))
        first = module.read(x[:, :2], bank)
        second = module.read(x[:, :2] + 3, bank)
        self.assertEqual(module.bank_builds, 1); self.assertEqual(module.reads, 2)
        self.assertFalse(torch.equal(first, second))

    def test_empty_bank_is_rejected(self):
        with self.assertRaises(ValueError):
            CompressRouteAttention(32, 4).build(torch.zeros(1, 3, 32), torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1, 32))


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_content_dedup_retains_addresses_and_multiplicity(self):
        store = ObservationStore(16)
        image = b"x" * 16 + b"y" * 16 + b"x" * 16
        observation = store.collect(image, [Frame("stdout", b"x" * 16)] * 5)
        self.assertEqual(len(observation.payloads), 2)
        self.assertEqual(observation.instances, 8)
        self.assertEqual(observation.addresses[:3], [0, 16, 32])
        self.assertEqual(observation.unique_ids[:3], [0, 1, 0])
        self.assertEqual(observation.total_payload_bytes, 128)
        self.assertEqual(observation.encoded_payload_bytes, 32)

    def test_changed_content_refreshes_and_addresses_remain_distinct(self):
        store = ObservationStore(16); observer = GreenObserver(32, 16, 4, 2, 2).eval()
        first = store.collect(b"a" * 32)
        with torch.inference_mode():
            features, _ = observer.encode(first)
            self.assertEqual(observer.encoded_chunks, 1)
            self.assertFalse(torch.equal(features[0, 0], features[0, 1]))
            observer.encode(first)
            self.assertEqual(observer.encoded_chunks, 1)
            changed = store.collect(b"b" + b"a" * 31)
            observer.encode(changed)
            self.assertEqual(observer.encoded_chunks, 2)
        self.assertNotEqual(first.image_sha256, changed.image_sha256)

    def test_training_and_weight_load_invalidate_feature_cache(self):
        observer = GreenObserver(32, 16, 4, 2, 2).eval(); obs = ObservationStore(16).collect(b"a" * 16)
        with torch.inference_mode(): observer.encode(obs)
        self.assertTrue(observer._content_cache)
        observer.train(); self.assertFalse(observer._content_cache)
        observer.eval()
        with torch.inference_mode(): observer.encode(obs)
        observer.load_state_dict(observer.state_dict())
        self.assertFalse(observer._content_cache)

    def test_capacity_is_not_silent_truncation(self):
        with self.assertRaises(ValueError):
            ObservationStore(16, 2).collect(b"a" * 48)


if __name__ == "__main__":
    unittest.main()
