import unittest

import torch

from lightnest.any2any import ArtifactCore, ArtifactRequest
from lightnest.context import NeuralNotebook


class ContextTests(unittest.TestCase):
    def test_notebook_keeps_learned_slot_budget_and_gradients(self):
        torch.manual_seed(4)
        notebook = NeuralNotebook(width=16, slots=4)
        evidence = torch.randn(2, 7, 16, requires_grad=True)
        state = notebook.compress(evidence, hard=False)
        self.assertEqual(state["context"].shape, (2, 16))
        self.assertEqual(state["budget_logits"].shape, (2, 5))
        state["context"].square().mean().backward()
        self.assertGreater(float(notebook.slot_keys.grad.abs().sum()), 0)
        self.assertGreater(float(notebook.budget[-1].weight.grad.abs().sum()), 0)

    def test_any2any_hierarchical_capacity_exceeds_local_window(self):
        model = ArtifactCore(width=32, heads=4, max_bytes=1 << 20, sequence_limit=4096)
        raw = bytes(range(256)) * 24
        request = ArtifactRequest("raw", "understand")
        with self.assertRaises(ValueError):
            model.encode(raw, request)
        encoded = model.encode_chunks(raw, request, chunk_size=256, max_chunks=8)
        self.assertEqual(encoded.shape, (1, 8, 32))


if __name__ == "__main__":
    unittest.main()
