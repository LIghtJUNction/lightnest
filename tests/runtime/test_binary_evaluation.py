from pathlib import Path
import tempfile
import unittest

from lightnest.runtime.execute import ExecutableRunner
from tools.evaluate_binary import evaluate


class BinaryEvaluationTests(unittest.TestCase):
    def test_real_binary_correct_mismatch_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ExecutableRunner(Path(directory))
            suite = {"name": "cat", "cases": [
                {"input": {"args": [], "stdin": "410a"},
                 "expected": {"returncode": 0, "stdout": "410a", "stderr": ""}},
                {"input": {"args": [], "stdin": "4200"},
                 "expected": {"returncode": 0, "stdout": "4200", "stderr": ""}},
            ]}
            good = evaluate(Path('/usr/bin/cat').read_bytes(), suite, runner)
            self.assertTrue(good['all_passed'])
            wrong = evaluate(Path('/usr/bin/true').read_bytes(), suite, runner)
            self.assertTrue(wrong['elf']['parsed'])
            self.assertEqual(wrong['passed_cases'], 0)
            invalid = evaluate(b'bad', suite, runner)
            self.assertFalse(invalid['elf']['parsed'])
            self.assertEqual(invalid['cases'][0]['actual']['status'], 'invalid_artifact')

    def test_empty_suite_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(b'x', {'name': 'empty', 'cases': []}, None)
