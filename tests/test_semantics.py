import tempfile
import unittest
from pathlib import Path

import torch

from lightnest.any2any import ArtifactCore
from lightnest.semantics import examples, match, metrics


class SemanticTests(unittest.TestCase):
    def test_labels_and_comments_do_not_survive_in_source_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'module.py'
            path.write_text('''def process(value):
    """Return the unique secret label for this example."""
    # Comment contains the same unique secret label.
    def nested(item):
        """Another secret label that must be removed."""
        return item * 2
    return nested(value) + 1
''')
            records = examples(path, root, 'external')
        self.assertGreater(len(records), 0)
        self.assertTrue(all('secret' not in row['code'] for row in records))
        self.assertIn('unique secret label', records[0]['description'])

    def test_tied_scores_cannot_report_perfect_retrieval(self):
        result = metrics(torch.zeros(8, 8))
        self.assertEqual(result['recall_at_1'], 1 / 8)
        self.assertEqual(result['recall_at_5'], 5 / 8)
        self.assertLess(result['mrr'], .5)

    def test_removing_first_docstring_does_not_promote_another_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'module.py'
            path.write_text('''def process(value):
    """Return the first secret explanation for this example."""
    """This second secret explanation must also disappear."""
    return value * 2 + 1
''')
            records = examples(path, root, 'external')
        self.assertEqual(len(records), 1)
        self.assertNotIn('secret', records[0]['code'])

    def test_match_reads_raw_input_without_a_source_parser(self):
        model = ArtifactCore(width=32, heads=4, max_bytes=256).eval()
        result = match(model, b'\x00\xffarbitrary input', ['first description', 'second description'])
        self.assertEqual({row['description'] for row in result}, {'first description', 'second description'})
        self.assertTrue(all(-1.001 <= row['score'] <= 1.001 for row in result))
        with self.assertRaises(ValueError):
            match(model, b'x' * 257, ['description'])

    def test_chunk_compression_retains_evidence_at_the_end(self):
        torch.manual_seed(29)
        model = ArtifactCore(width=32, heads=4, max_bytes=1024, semantic_chunk_size=8).eval()
        with torch.inference_mode():
            a = model.semantic_embedding([b'a' * 1024])
            b = model.semantic_embedding([b'a' * 1023 + b'b'])
        self.assertGreater(float((a - b).abs().max()), 1e-5)

    def test_chunk_padding_does_not_change_short_file(self):
        torch.manual_seed(31)
        model = ArtifactCore(width=32, heads=4, max_bytes=256, semantic_chunk_size=8).eval()
        with torch.inference_mode():
            alone = model.semantic_embedding([b'def increase(value): return value + 1'])
            batch = model.semantic_embedding([b'def increase(value): return value + 1', bytes(range(256))])
        self.assertTrue(torch.allclose(alone[0], batch[0], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
