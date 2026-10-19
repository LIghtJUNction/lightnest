"""Tool tests without Lean or model weights; full proof checks run in Lean CI."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.formal_contracts import (aligned_comparisons, compare_evidence,
                                    read_evidence, retention_gate, schedule_cost)
from tools.check_lean import (ROOT, check_sources, conformance_source, verify,
                             without_comments_and_strings)
from tools.check_source_tree import check


def record(name, success=False, wrong=False):
    return {'id':name, 'initial_sha256':hashlib.sha256(name.encode()).hexdigest(),
            'strict':success, 'false_edit':wrong}


class RetentionTests(unittest.TestCase):
    def test_keeps_success_and_allows_improvement(self):
        a = [record('a', True), record('b')]
        b = [record('b', True), record('a', True)]
        got = retention_gate(a, b)
        self.assertTrue(got['retention_passed'])
        self.assertEqual(got['added_successes'], 1)
        self.assertFalse(got['model_promoted'])

    def test_higher_total_does_not_hide_regression(self):
        a = [record('a', True), record('b'), record('c')]
        b = [record('a'), record('b', True), record('c', True)]
        self.assertFalse(retention_gate(a, b)['retention_passed'])

    def test_wrong_edit_rejected_even_when_parent_failed(self):
        self.assertFalse(retention_gate([record('a')], [record('a', wrong=True)])['retention_passed'])

    def test_empty_lists_fail_closed(self):
        with self.assertRaises(ValueError):
            retention_gate([], [])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            aligned_comparisons([record('a')] * 2, [record('a')] * 2)

    def test_missing_ids_rejected(self):
        with self.assertRaises(ValueError):
            aligned_comparisons([record('a')], [record('b')])

    def test_changed_input_rejected(self):
        b = record('a') | {'initial_sha256':'0' * 64}
        with self.assertRaises(ValueError):
            aligned_comparisons([record('a')], [b])

    def test_scores_and_string_booleans_rejected(self):
        for value in (0, 1, 0.99, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                aligned_comparisons([record('a')], [record('a') | {'strict':value}])

    def test_inconsistent_success_rejected(self):
        with self.assertRaises(ValueError):
            aligned_comparisons([record('a')], [record('a', True, True)])

    def test_provenance_binding_required(self):
        a = {'records':[record('a')]}
        with self.assertRaises(ValueError):
            compare_evidence(a, a)
        for key in ('dataset_sha256', 'evaluator_sha256', 'protocol_sha256'):
            a[key] = 'a' * 64
        self.assertTrue(compare_evidence(a, a)['retention_passed'])
        b = copy.deepcopy(a); b['evaluator_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            compare_evidence(a, b)

    def test_read_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'evidence.json'
            p.write_text(json.dumps({'records':[record('a')]}))
            with self.assertRaises(ValueError):
                read_evidence(p)


class ScheduleTests(unittest.TestCase):
    def test_real_schedule(self):
        self.assertEqual(schedule_cost([4, 8, 16, 32, 64]), 64)

    def test_bad_schedules_fail(self):
        for values in ([], [0, 64], [4, 4, 64], [8, 4, 64], [4, 8], [65], [True, 64]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                schedule_cost(values)

    def test_bad_maximum(self):
        for maximum in (0, -1, True, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                schedule_cost([64], maximum)


class ProofToolTests(unittest.TestCase):
    def test_stated_proof_inventory(self):
        self.assertGreaterEqual(check_sources(ROOT / 'formal'), 26)

    def test_nested_comments_and_strings(self):
        source = '/- sorry /- axiom -/ admit -/\ntheorem good : True := by trivial -- unsafe\n"sorry"'
        clean = without_comments_and_strings(source)
        self.assertNotIn('sorry', clean)
        self.assertNotIn('axiom', clean)
        self.assertIn('theorem good', clean)
        self.assertEqual(source.count('\n'), clean.count('\n'))

    def test_unclosed_comment_fails(self):
        with self.assertRaises(ValueError):
            without_comments_and_strings('/- forgotten')

    def test_placeholder_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            formal = Path(td) / 'formal'
            shutil.copytree(ROOT / 'formal', formal, ignore=shutil.ignore_patterns('.lake'))
            p = formal / 'LightNest/Patch.lean'
            p.write_text(p.read_text() + '\ntheorem escaped : True := by sorry\n')
            with self.assertRaisesRegex(ValueError, 'forbidden proof escape'):
                check_sources(formal)

    def test_unimported_proof_fails(self):
        with tempfile.TemporaryDirectory() as td:
            formal = Path(td) / 'formal'
            shutil.copytree(ROOT / 'formal', formal, ignore=shutil.ignore_patterns('.lake'))
            (formal / 'LightNest/Lost.lean').write_text('import Std\n')
            with self.assertRaisesRegex(ValueError, 'cover every proof'):
                check_sources(formal)

    def test_no_lean_cannot_report_success(self):
        with patch('tools.check_lean.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'No Lean proof was checked'):
                verify()

    def test_conformance_generation_uses_current_guard(self):
        source, counts = conformance_source(ROOT)
        self.assertGreater(counts['patch'], 300)
        self.assertEqual(sum(counts.values()), source.count(':= by decide'))
        self.assertIn('Patch.checkedPatch', source)
        self.assertIn('Compute.scheduleCost', source)
        self.assertNotIn('native_decide', source)

    def test_lean_sources_allowed_in_source_only_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'Contract.lean').write_text('theorem test : True := by trivial\n')
            (root / 'lean-toolchain').write_text('leanprover/lean4:v4.35.0\n')
            self.assertEqual(check(root)[1], [])

    def test_tracked_lean_build_product_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            p = root / '.lake/build/Fake.olean'
            p.parent.mkdir(parents=True); p.write_bytes(b'not a proof\x00')
            subprocess.run(['git', '-C', str(root), 'add', '-f', '.lake'], check=True)
            errors = check(root)[1]
            self.assertTrue(any('generated-data' in e for e in errors))
            self.assertTrue(any('forbidden extension' in e for e in errors))

    def test_binary_payload_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'fake.lean').write_bytes(b'not source\x00')
            self.assertTrue(check(root)[1])


if __name__ == '__main__':
    unittest.main()
