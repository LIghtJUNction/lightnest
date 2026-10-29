"""Data-free regression checks for the v9 finite-catalogue and DFA components.

The complete v9 historical evaluation (musl splits, trained models, 69,924 native
cases) lives in the separately hash-pinned evidence bundle, not in the git tree.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lightnest_rgb.risk_planner import (choose, entropy2_tree, entropy2_values,
                                        pair_features, query_values, tree_success)
from lightnest_rgb.automata import (DFA, TableLearner, emit_demo_elf, emit_dfa,
                                    equivalent, utf8_reference, verify_relation)


class DemoEntryPointTests(unittest.TestCase):
    def test_documented_script_starts_outside_the_checkout(self):
        script = Path(__file__).resolve().parents[1] / 'tools/validation_v9_demo.py'
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(script), '--help'], cwd=temporary,
                                    env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--out', result.stdout)


def brute(h, p, x, z0, z1):
    transcript = {}
    for i in range(len(p)):
        first = int(h[i, x])
        second = int(h[i, z0 if first == 0 else z1])
        transcript.setdefault((first, second), []).append(i)
    return sum(max(p[j] for j in indices) for indices in transcript.values())


class PlanningV9Tests(unittest.TestCase):
    def test_exhaustive_two_queries(self):
        rng = np.random.default_rng(95001)
        for _ in range(32):
            h = rng.integers(0, 2, (8, 5), dtype=np.uint8)
            p = rng.dirichlet(np.ones(8) * .3)
            _, values, _ = query_values(h, p)
            for x in range(5):
                exact = max(brute(h, p, x, a, b) for a in range(5) for b in range(5))
                self.assertAlmostEqual(values[x], exact, places=12)
            self.assertAlmostEqual(tree_success(h, p, 'risk2')['weighted'], values.max())

    def test_horizon_matched_entropy(self):
        rng = np.random.default_rng(95010)
        for _ in range(18):
            h = rng.integers(0, 2, (8, 4), dtype=np.uint8)
            p = rng.dirichlet(np.ones(8) * .3)
            vals = entropy2_values(h, p)
            for x in range(4):
                exact = []
                for y in range(4):
                    for z in range(4):
                        mass = {}
                        for i in range(8):
                            first = int(h[i, x]); second = int(h[i, y if first == 0 else z])
                            mass[(first, second)] = mass.get((first, second), 0) + p[i]
                        exact.append(-sum(q * np.log2(q) for q in mass.values()))
                self.assertAlmostEqual(vals[x], max(exact), places=12)
            self.assertLessEqual(entropy2_tree(h, p)['weighted'],
                                 tree_success(h, p, 'risk2')['weighted'] + 1e-12)

    def test_local_entropy_vs_success_objective(self):
        h = np.zeros((8, 7), np.uint8)
        h[0, 1] = 1
        h[2:7, 0] = 1
        h[2:7, 2:] = np.eye(5, dtype=np.uint8)
        h[7] = 1
        p = np.array([.3, .2, .1, .1, .1, .1, .1, 0.])
        self.assertEqual(choose(h, p, 'entropy', 1), 0)
        self.assertEqual(choose(h, p, 'risk1', 1), 1)

    def test_constant_and_invalid_candidate_sets(self):
        h = np.zeros((8, 16), np.uint8)
        p = np.arange(1, 9, dtype=float)
        p /= p.sum()
        for policy in ('random', 'entropy', 'risk1', 'risk2'):
            self.assertAlmostEqual(tree_success(h, p, policy)['weighted'], p.max())
        for p in (np.zeros(8), np.ones(7), np.full(8, np.nan)):
            with self.assertRaises(ValueError):
                query_values(h, p)

    def test_fixed_feature_shape(self):
        h = np.zeros((8, 16), dtype=np.uint8)
        p = np.ones(8) / 8
        self.assertEqual(pair_features(h, p).shape, (16, 16, 24))


class AutomataV9Tests(unittest.TestCase):
    def test_rfc_boundaries_against_python(self):
        dfa = utf8_reference()
        samples = [b'', b'\x00', b'abc\x00', b'\xc2\x80', b'\xe0\xa0\x80',
                   b'\xf4\x8f\xbf\xbf', b'\xc0\x80', b'\xed\xa0\x80',
                   b'\xf4\x90\x80\x80', b'\xe0\x9f\xbf', b'\xc2']
        for data in samples:
            try:
                data.decode('utf-8', 'strict')
                want = 1
            except UnicodeDecodeError:
                want = 0
            self.assertEqual(dfa.run(data), want)

    def test_relation_and_state_permutation(self):
        ref = utf8_reference()
        rng = np.random.default_rng(95003)
        perm = rng.permutation(9)
        inv = np.argsort(perm)
        cand = DFA(perm[ref.transition[inv]], ref.accept[inv], int(perm[ref.start]))
        result = equivalent(ref, cand)
        self.assertTrue(result['equivalent'])
        self.assertTrue(verify_relation(ref, cand, result['relation']))
        self.assertFalse(verify_relation(ref, cand, result['relation'][:-1]))

    def test_shortest_witness_and_perturbation(self):
        ref = utf8_reference()
        trans = ref.transition.copy()
        trans[0, 0xc2] = 8
        bad = DFA(trans, ref.accept)
        result = equivalent(ref, bad)
        self.assertFalse(result['equivalent'])
        witness = bytes.fromhex(result['witness_hex'])
        self.assertEqual(len(witness), 2)
        self.assertNotEqual(ref.run(witness), bad.run(witness))
        for x in range(256):
            self.assertEqual(ref.run(bytes([x])), bad.run(bytes([x])))

    def test_empty_witness(self):
        ref = utf8_reference()
        accept = ref.accept.copy()
        accept[0] ^= 1
        self.assertEqual(equivalent(ref, DFA(ref.transition, accept))['witness_hex'], '')

    def test_learns_toy_dfa_without_hidden_transitions(self):
        t = np.zeros((2, 256), dtype=np.uint8)
        for s in range(2):
            t[s] = s ^ (np.arange(256) & 1)
        ref = DFA(t, np.array([1, 0], np.uint8))
        learner = TableLearner(lambda w: sum(x & 1 for x in w) % 2 == 0,
                               lambda cand: equivalent(ref, cand))
        got = learner.learn()
        self.assertTrue(equivalent(ref, got)['equivalent'])
        self.assertEqual(len(got.accept), 2)

    def test_invalid_models(self):
        with self.assertRaises(ValueError):
            DFA(np.zeros((1, 255), dtype=np.uint8), np.ones(1, dtype=np.uint8))
        with self.assertRaises(ValueError):
            DFA(np.ones((1, 256), dtype=np.uint8), np.ones(1, dtype=np.uint8))

    def test_encoder_layout_determinism(self):
        machine = utf8_reference()
        code = emit_dfa(machine)
        self.assertEqual(code, emit_dfa(machine))
        elf = emit_demo_elf(code, b'test')
        self.assertEqual(elf[:4], b'\x7fELF')
        self.assertEqual(struct.unpack_from('<Q', elf, 24)[0], 0x400000 + 120)
        self.assertEqual(struct.unpack_from('<Q', elf, 96)[0], len(elf))


class NativeV9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('gcc'):
            raise RuntimeError('GCC is required by the Linux native runner tests')
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.runner = root / 'runner'
        source = Path(__file__).resolve().parents[1] / 'tools/behavior_native_runner.c'
        subprocess.run(['gcc', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        str(source), '-o', str(cls.runner)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_native(self, code, cases):
        folder = Path(self.tmp.name)
        fn = folder / 'function.bin'
        inp = folder / 'cases.bin'
        fn.write_bytes(code)
        inp.write_bytes(struct.pack('<I', len(cases)) +
                        b''.join(struct.pack('<I', len(x)) + x for x in cases))
        return subprocess.run([str(self.runner), str(fn), str(inp)],
                              capture_output=True, timeout=3)

    def test_compiled_utf8_dfa_against_python(self):
        cases = [b''] + [bytes([x]) for x in range(256)]
        cases += [bytes([0xc2, x]) for x in range(256)]
        cases += [b'a' * 131072, b'\xe0\xa0\x80' * 2048, b'\xff' * 4096]
        # The runner caps each call at 512 cases; test in two batches.
        code = emit_dfa(utf8_reference())
        for batch in (cases[:256], cases[256:]):
            out = self.run_native(code, batch)
            self.assertEqual(out.returncode, 0, out.stderr)
            got = struct.unpack('<' + 'Q' * len(batch), out.stdout)
            expected = []
            for value in batch:
                try:
                    value.decode('utf-8', 'strict')
                    expected.append(1)
                except UnicodeDecodeError:
                    expected.append(0)
            self.assertEqual(got, tuple(expected))

    def test_runner_rejects_forbidden_syscall(self):
        self.assertNotEqual(self.run_native(bytes.fromhex('b8270000000f05c3'), [b'']).returncode, 0)

    def test_runner_rejects_guard_read_and_input_write(self):
        self.assertNotEqual(self.run_native(bytes.fromhex('0fb607c3'), [b'']).returncode, 0)
        self.assertNotEqual(self.run_native(bytes.fromhex('c60700c3'), [b'A']).returncode, 0)


if __name__ == '__main__':
    unittest.main()
