import json
import unittest

import torch
from torch.nn import functional as F

from lightnest_rgb.any2any import ArtifactCore, semantic_similarity
from lightnest_rgb.byte_tokenizer import ByteBPE


class ByteTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ByteBPE.train([b'return value + 1\n' * 8, b'return a smaller value\n' * 8], vocab_size=288)

    def model(self):
        return ArtifactCore(width=32, heads=4, max_bytes=512, sequence_limit=512,
                            semantic_bpe=self.tokenizer.serialized)

    def test_unseen_binary_bytes_round_trip_without_unknown_tokens(self):
        raw = bytes(range(256)) + b'\x00\xff\x00new artifact'
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(raw)), raw)

    def test_normalization_and_truncation_are_rejected(self):
        config = json.loads(self.tokenizer.serialized)
        config['normalizer'] = {'type': 'Lowercase'}
        with self.assertRaises(ValueError):
            ByteBPE(json.dumps(config))

    def test_padding_does_not_change_scores_or_language(self):
        torch.manual_seed(5)
        model = self.model().eval()
        with torch.inference_mode():
            short = model.semantic_tokens([b'return value + 1'])
            batch = model.semantic_tokens([b'return value + 1', bytes(range(256))])
            query = model.semantic_tokens([b'return a smaller value'])
            a = semantic_similarity(short, query)[0]
            b = semantic_similarity(batch, query)[0]
            self.assertTrue(torch.allclose(a, b, atol=1e-5))
            previous = torch.tensor([[self.tokenizer.bos]])
            first, _ = model.language_forward(short, previous)
            both, _ = model.language_forward(batch, previous.expand(2, -1))
            self.assertTrue(torch.allclose(first[0], both[0], atol=1e-5))

    def test_language_prediction_cannot_read_future_target_tokens(self):
        model = self.model().eval()
        with torch.inference_mode():
            memory = model.semantic_tokens([b'return value + 1'])
            a, _ = model.language_forward(memory, torch.tensor([[self.tokenizer.bos, 10, 20]]))
            b, _ = model.language_forward(memory, torch.tensor([[self.tokenizer.bos, 80, 90]]))
            self.assertTrue(torch.allclose(a[:, 0], b[:, 0], atol=1e-5))

    def test_binary_language_loss_updates_the_artifact_encoder(self):
        model = self.model()
        raw = bytes(range(64))
        memory = model.semantic_tokens([raw])
        target = torch.tensor([[self.tokenizer.bos, 17, 23]])
        output, _ = model.language_forward(memory, target)
        F.nll_loss(output.flatten(0, 1), torch.tensor([17, 23, self.tokenizer.eos])).backward()
        self.assertGreater(float(model.encoder.layers[0].linear1.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.language_copy.weight.grad.abs().sum()), 0)

    def test_complete_input_beyond_local_window_and_tail_are_observed(self):
        torch.manual_seed(6)
        model = ArtifactCore(width=32, heads=4, max_bytes=4096, sequence_limit=256,
                             semantic_bpe=self.tokenizer.serialized).eval()
        raw = bytes(range(256)) * 3
        with torch.inference_mode():
            original = model.semantic_tokens([raw])
            changed = model.semantic_tokens([raw[:-1] + b'\x00'])
        self.assertGreater(int(original['valid'].sum()), model.sequence_limit)
        self.assertEqual(self.tokenizer.decode(original['tokens'][0, original['valid'][0]].tolist()), raw)
        self.assertGreater(float((original['features'][:, -1] - changed['features'][:, -1]).abs().max()), .001)


if __name__ == '__main__':
    unittest.main()
