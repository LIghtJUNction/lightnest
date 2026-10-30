"""Primitive tests only: no model training, ELF execution, or promotion claim."""
import io
import unittest

import torch
from torch.nn import functional as F

from archive.research.exact_core import ExactBitRAM, FixedWordAdder, decode_words, encode_words


class ExactCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_roundtrip_integer_precision_boundaries(self):
        words = torch.tensor([0, 1, 255, 256, 257, 2**24 + 1, 2**31, 2**32 - 1])
        for dtype in (torch.float32, torch.bfloat16):
            self.assertTrue(torch.equal(words, decode_words(encode_words(words, dtype=dtype))))

    def test_encoding_rejects_out_of_range(self):
        for words in (torch.tensor([-1]), torch.tensor([2**32]), torch.empty(0, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                encode_words(words)
        with self.assertRaises(ValueError):
            encode_words(torch.tensor([1.0]))

    def test_ram_reads_before_writes_and_never_mutates_input(self):
        for dtype in (torch.float32, torch.bfloat16):
            ram = ExactBitRAM(4, 32)
            state = ram.empty(2, dtype=dtype)
            key = F.one_hot(torch.tensor([0, 3]), 4).to(dtype)
            value = encode_words(torch.tensor([2**32 - 1, 2**24 + 1]), dtype=dtype)
            written, old = ram(state, key, value, torch.ones(2, 1, dtype=dtype))
            self.assertEqual(int(state.count_nonzero()), 0)
            self.assertEqual(int(old.count_nonzero()), 0)
            unchanged, read = ram(written, key, torch.zeros_like(value), torch.zeros(2, 1, dtype=dtype))
            self.assertTrue(torch.equal(read, value))
            self.assertTrue(torch.equal(unchanged, written))
            self.assertEqual(int(written[0, 1:].count_nonzero()), 0)
            self.assertEqual(int(written[1, :3].count_nonzero()), 0)

    def test_ram_fork_is_independent(self):
        ram = ExactBitRAM(2, 8)
        parent = ram.empty()
        child = parent.clone()
        key = torch.tensor([[1., 0.]])
        child, _ = ram(child, key, torch.ones(1, 8), torch.ones(1, 1))
        self.assertEqual(int(parent.count_nonzero()), 0)
        self.assertEqual(int(child.count_nonzero()), 8)

    def test_ram_rejects_soft_zero_and_multirow_keys(self):
        ram = ExactBitRAM(2, 8)
        for key in (torch.tensor([[0.5, 0.5]]), torch.zeros(1, 2), torch.ones(1, 2),
                    torch.tensor([[float('nan'), 1.]])):
            with self.assertRaises(ValueError):
                ram(ram.empty(), key, torch.ones(1, 8), torch.ones(1, 1))

    def test_ram_rejects_soft_gate_or_state(self):
        ram = ExactBitRAM(2, 8)
        key = torch.tensor([[1., 0.]])
        with self.assertRaises(ValueError):
            ram(ram.empty(), key, torch.ones(1, 8), torch.tensor([[0.9]]))
        with self.assertRaises(ValueError):
            ram(ram.empty() + 0.5, key, torch.ones(1, 8), torch.ones(1, 1))

    def test_ram_rejects_shape_and_dtype_mismatch(self):
        ram = ExactBitRAM(2, 8)
        for key in (torch.zeros(1, 3), torch.tensor([[1., 0.]], dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):
                ram(ram.empty(), key, torch.ones(1, 8), torch.ones(1, 1))

    def test_one_bit_full_adder_truth_table(self):
        a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        b = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        c = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        adder = FixedWordAdder(1)
        result, carry = adder(encode_words(a, 1), encode_words(b, 1), c[:, None].float())
        self.assertTrue(torch.equal(decode_words(result), (a + b + c) % 2))
        self.assertTrue(torch.equal(carry[:, 0].long(), (a + b + c) // 2))

    def test_eight_bit_exhaustive_with_both_carry_inputs(self):
        a = torch.arange(256).repeat_interleave(256)
        b = torch.arange(256).repeat(256)
        for dtype in (torch.float32, torch.bfloat16):
            adder = FixedWordAdder(8).to(dtype=dtype)
            for carry_in in (0, 1):
                expected = a + b + carry_in
                result, carry = adder(encode_words(a, 8, dtype=dtype), encode_words(b, 8, dtype=dtype),
                                      torch.full((len(a), 1), carry_in, dtype=dtype))
                self.assertTrue(torch.equal(decode_words(result), expected % 256))
                self.assertTrue(torch.equal(carry[:, 0].long(), expected // 256))

    def test_uint32_edges_and_random_operands(self):
        generator = torch.Generator().manual_seed(20261029)
        edges = torch.tensor([0, 1, 255, 257, 2**24 + 1, 2**31, 2**32 - 1])
        a = torch.cat((edges.repeat_interleave(len(edges)), torch.randint(2**32, (4096,), generator=generator)))
        b = torch.cat((edges.repeat(len(edges)), torch.randint(2**32, (4096,), generator=generator)))
        for dtype in (torch.float32, torch.bfloat16):
            adder = FixedWordAdder(32).to(dtype=dtype)
            for carry_in in (0, 1):
                expected = a + b + carry_in
                result, carry = adder(encode_words(a, dtype=dtype), encode_words(b, dtype=dtype),
                                      torch.full((len(a), 1), carry_in, dtype=dtype))
                self.assertTrue(torch.equal(decode_words(result), expected % 2**32))
                self.assertTrue(torch.equal(carry[:, 0].long(), expected // 2**32))

    def test_buffers_are_not_trainable_and_checkpoint_roundtrips(self):
        adder = FixedWordAdder(32)
        self.assertEqual(list(adder.parameters()), [])
        self.assertEqual(list(ExactBitRAM().parameters()), [])
        stream = io.BytesIO()
        torch.save(adder.state_dict(), stream)
        stream.seek(0)
        restored = FixedWordAdder(32)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        for name, value in adder.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]))

    def test_adder_rejects_nonbinary_inputs(self):
        adder = FixedWordAdder(8)
        with self.assertRaises(ValueError):
            adder(torch.full((1, 8), 0.5), torch.zeros(1, 8))
        with self.assertRaises(ValueError):
            adder(torch.zeros(1, 8), torch.zeros(1, 8), torch.tensor([[2.]]))


    def test_addition_stress_tail_and_matching_dtype_inputs(self):
        from archive.research.tools.stress_exact_core import addition_probe
        results = [addition_probe(4097, 91, dtype) for dtype in (torch.float32, torch.bfloat16)]
        for result in results:
            self.assertEqual(result['cases'], 4097)
            self.assertEqual(result['mismatches'], 0)
        self.assertEqual(results[0]['input_sha256'], results[1]['input_sha256'])

    def test_feedback_cross_dtype_and_revisited_rows(self):
        from archive.research.tools.stress_exact_core import feedback_probe
        results = [feedback_probe(17, 92, dtype, lanes=3) for dtype in (torch.float32, torch.bfloat16)]
        for result in results:
            self.assertEqual(result['word_updates'], 51)
            for key in ('read_mismatches', 'result_or_carry_mismatches',
                        'final_word_mismatches', 'reference_corrections'):
                self.assertEqual(result[key], 0)
        self.assertEqual(results[0]['input_sha256'], results[1]['input_sha256'])
        self.assertEqual(results[0]['final_words_sha256'], results[1]['final_words_sha256'])

    def test_feedback_detects_corruption_without_reference_repair(self):
        from unittest.mock import patch
        from archive.research.tools.stress_exact_core import feedback_probe
        original = FixedWordAdder.forward

        def corrupted(module, a, b, carry_in=None):
            result, carry = original(module, a, b, carry_in)
            return 1 - result, carry

        with patch.object(FixedWordAdder, 'forward', corrupted):
            result = feedback_probe(17, 93, torch.float32, lanes=3)
        self.assertGreater(result['result_or_carry_mismatches'], 0)
        self.assertGreater(result['read_mismatches'], 0)
        self.assertGreater(result['final_word_mismatches'], 0)
        self.assertEqual(result['reference_corrections'], 0)


if __name__ == '__main__':
    unittest.main()
