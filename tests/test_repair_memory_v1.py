"""Source-only tests; no saved weights or evaluation dataset required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.compiled_v3.vm import NativeVM, Input, Output, compile_kernel, reference, pack
from experiments.compiled_v3.search import Request, repair, patched
from experiments.compiled_v3.run import make_mutants
from experiments.compiled_v3.corpus import source
from experiments.compiled_v3.inputs import random_inputs
from experiments.repair_memory_v1.repair import repair_bridge

MODES = ('single', 'random_pair', 'random_memory', 'bridge_only', 'bridge')


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.code, cls.entry, native = compile_kernel(source('prefix_scan', 11), cls.root/'kernel', 'O2')
        cls.vm = NativeVM(cls.root/'vm')
        cls.inputs = tuple(random_inputs(8732, 16))
        cls.expected = tuple(reference(native, cls.inputs))
        cls.tasks = make_mutants(cls.code, 12693, 6, 4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def request(self, task=None):
        return Request(self.code if task is None else bytes.fromhex(task['code']),
                       self.entry, self.inputs, self.expected)

    def test_clean_unchanged_in_all_modes(self):
        for mode in MODES:
            r = repair_bridge(self.request(), self.vm, mode=mode)
            self.assertEqual((r.code, r.status, r.tried, r.patches),
                             (self.code, 'unchanged', 0, ()))

    def test_single_phase_matches_original(self):
        for task in self.tasks:
            a = repair(self.request(task), self.vm, guided=True, budget=64)
            b = repair_bridge(self.request(task), self.vm, mode='single', budget=64)
            self.assertEqual((a.code, a.status, a.tried), (b.code, b.status, b.tried))

    def test_successful_single_is_preserved(self):
        for task in self.tasks:
            req = self.request(task)
            old = repair(req, self.vm, guided=True, budget=128)
            if old.status in ('candidate', 'unchanged'):
                for mode in MODES:
                    new = repair_bridge(req, self.vm, mode=mode, budget=128)
                    self.assertEqual((new.code, new.status, new.tried),
                                     (old.code, old.status, old.tried))

    def test_outputs_reconstruct_and_pass_full_visible_suite(self):
        for task in self.tasks:
            req = self.request(task)
            for mode in MODES:
                result = repair_bridge(req, self.vm, mode=mode, budget=128)
                self.assertLessEqual(result.tried, 128)
                self.assertLessEqual(len(result.patches), 2)
                self.assertLessEqual(result.kept, 8)
                code = req.code
                for change in result.patches:
                    code = patched(code, *change)
                self.assertEqual(code, result.code)
                if result.status == 'candidate':
                    self.assertLess(self.vm.check(code, req.entry, pack(req.cases, req.expected)), 0)

    def test_no_caller_mutation(self):
        before = (self.code, [bytes(x) for x in self.inputs], [bytes(x) for x in self.expected])
        for mode in MODES:
            repair_bridge(self.request(self.tasks[-1]), self.vm, mode=mode, budget=32)
        self.assertEqual(before, (self.code, [bytes(x) for x in self.inputs], [bytes(x) for x in self.expected]))

    def test_deterministic_replay(self):
        req = self.request(self.tasks[-1])
        self.assertEqual(repair_bridge(req, self.vm, budget=256), repair_bridge(req, self.vm, budget=256))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            repair_bridge(self.request(), self.vm, mode='guess')

    def test_invalid_budget_and_width(self):
        for name, value in [('budget', 0), ('budget', True), ('budget', 100001),
                            ('width', 0), ('width', True), ('width', 33)]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                repair_bridge(self.request(), self.vm, **{name:value})

    def test_partial_progress_is_not_final_acceptance(self):
        class FakeVM:
            work = [0, 0]
            def check(self, code, entry, packed):
                self.work[0] += 1
                return 0 if code[:4] == bytes(4) else (1 if code[4:] == bytes(4) else -1)
            def run(self, code, entry, inp, trace=False):
                self.work[0] += 1
                return Output(1, (type(inp.a))(), (type(inp.a))(), 0, 1), {0, 1}
        inp = Input.make([0]*32, 0, 0)
        expected = Output()
        req = Request(bytes(8), 0x1000, (inp, inp), (expected, expected))
        def options(code, coverage=None):
            return [(0, 1), (4, 1)] if code == bytes(8) else [(4, 1)]
        with patch('experiments.repair_memory_v1.repair.candidates', side_effect=options):
            limited = repair_bridge(req, FakeVM(), mode='bridge_only', budget=2)
            self.assertNotEqual(limited.status, 'candidate')
            self.assertEqual(limited.code, req.code)
            solved = repair_bridge(req, FakeVM(), mode='bridge_only', budget=3)
            self.assertEqual(solved.status, 'candidate')
            self.assertEqual(solved.patches, ((0, 1), (4, 1)))

if __name__ == '__main__':
    unittest.main()
