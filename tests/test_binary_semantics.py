import ast
import importlib.util
import marshal
import random
import unittest

import torch
from torch.nn import functional as F

from lightnest_rgb.any2any import semantic_similarity
from lightnest_rgb.binary_semantics import behavior_probes, compile_bytecode, mine_negatives, sample_batch, score_matrix


class BinarySemanticTests(unittest.TestCase):
    def test_retrieval_scores_do_not_depend_on_padding_or_block_size(self):
        torch.manual_seed(17)
        def memory(lengths, maximum):
            valid = torch.arange(maximum)[None] < torch.tensor(lengths)[:, None]
            weights = F.normalize(valid.float(), p=1, dim=1)
            return {'features': torch.randn(len(lengths), maximum, 8), 'valid': valid, 'weights': weights}
        left, right = memory([2, 7, 3, 4], 10), memory([3, 2, 5], 8)
        expected = semantic_similarity(left, right)
        self.assertTrue(torch.allclose(expected, score_matrix(left, right, block=2), atol=1e-6))
    def test_compiled_input_contains_executable_behavior_without_documentation(self):
        source = 'def subtract(left, right):\n    return left - right\n'
        raw = compile_bytecode(source)
        self.assertEqual(raw[:4], importlib.util.MAGIC_NUMBER)
        code = marshal.loads(raw[16:])
        namespace = {}
        exec(code, {'__builtins__': {}}, namespace)
        self.assertEqual(namespace['subtract'](7, 2), 5)
        self.assertEqual(namespace['subtract'].__doc__, None)
        self.assertEqual(code.co_filename, '<artifact>')

    def test_target_documentation_cannot_leak_through_compilation(self):
        with self.assertRaises(ValueError):
            compile_bytecode('def f():\n    """Return a secret explanation."""\n    return 1\n')

    def test_hard_negative_batch_has_distinct_positive_labels(self):
        rows = [{'description': f'Return the cached result for item {i}', 'code_sha256': str(i)} for i in range(12)]
        neighbors = mine_negatives(rows)
        chosen = sample_batch(random.Random(1), len(rows), 8, neighbors)
        self.assertEqual(len(chosen), 8)
        self.assertEqual(len(set(chosen)), 8)
        self.assertTrue(all(i not in row for i, row in enumerate(neighbors)))

    def test_counterfactuals_have_same_word_counts_but_different_executed_behavior(self):
        probes = behavior_probes()
        self.assertEqual(len(probes), 32)
        for probe in probes:
            a, b = probe['programs']
            self.assertNotEqual(a['outputs'], b['outputs'])
            def names(source):
                return sorted(node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name))
            self.assertEqual(names(a['source']), names(b['source']))


if __name__ == '__main__':
    unittest.main()
