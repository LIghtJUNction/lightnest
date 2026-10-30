"""Behavior-space and loop-encoder tests; no external weights needed."""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from archive.research.behavior_space import BehaviorSpace, InconsistentEvidence, fit_penalty, transitions
from archive.research.loop_lift import emit_demo_elf, emit_scan


class PosteriorTests(unittest.TestCase):
    def test_small_exhaustive_posteriors(self):
        rng = np.random.default_rng(701)
        for n in range(2, 9):
            allrows = np.array(list(itertools.product((0, 1), repeat=n)), np.uint8)
            ts = np.count_nonzero(allrows != np.roll(allrows, 1, axis=1), axis=1)
            for cap in range(n+1):
                for beta in (0.0, 1.3):
                    space = BehaviorSpace(n, cap, beta)
                    for _ in range(3):
                        known = rng.choice([-1, 0, 1], size=n, p=[.6, .2, .2])
                        ok = (ts <= cap) & np.all((known < 0) | (allrows == known), axis=1)
                        if not ok.any():
                            with self.assertRaises(InconsistentEvidence):
                                space.posterior(known)
                            with self.assertRaises(InconsistentEvidence):
                                space.decode(known)
                            continue
                        post = space.posterior(known)
                        w = np.exp(-beta*ts[ok])
                        np.testing.assert_allclose(post.probability_one, w @ allrows[ok]/w.sum(),
                                                   atol=2e-12)
                        self.assertAlmostEqual(post.log2_behavior_count, math.log2(int(ok.sum())))
                        support = np.stack([(allrows[ok] == b).any(0) for b in (0, 1)], axis=1)
                        np.testing.assert_array_equal(post.possible, support)
                        pred = space.decode(known)
                        self.assertTrue(np.all((known < 0) | (pred == known)))
                        self.assertEqual(transitions(pred), int(ts[ok].min()))

    def test_ring_symmetries(self):
        known = np.array([-1, 0, 0, -1, 1, -1, -1, 0])
        s = BehaviorSpace(8, 4, 1.7)
        p = s.posterior(known).probability_one
        for shift in range(8):
            np.testing.assert_allclose(s.posterior(np.roll(known, shift)).probability_one,
                                       np.roll(p, shift))
        np.testing.assert_allclose(s.posterior(known[::-1]).probability_one, p[::-1])

    def test_no_unconditional_identification(self):
        s = BehaviorSpace(8, 8, 0)
        known = np.array([0, 1, -1, -1, -1, -1, -1, -1])
        p = s.posterior(known)
        self.assertEqual(p.log2_behavior_count, 6)
        self.assertFalse(p.conditional_unique)

    def test_query_never_reads_known(self):
        s = BehaviorSpace(8, 4, 1)
        known = np.full(8, -1)
        truth = np.array([0, 0, 1, 1, 0, 0, 0, 0])
        for _ in range(8):
            p = s.posterior(known)
            q = p.choose_query()
            if q is None:
                break
            self.assertEqual(known[q], -1)
            known[q] = truth[q]
        np.testing.assert_array_equal(s.decode(known), truth)

    def test_contradiction_not_silently_dropped(self):
        with self.assertRaises(InconsistentEvidence):
            BehaviorSpace(8, 0).posterior(np.array([0, 1, -1, -1, -1, -1, -1, -1]))

    def test_unique_is_class_conditional(self):
        p = BehaviorSpace(8, 0).posterior(np.array([0, -1, -1, -1, -1, -1, -1, -1]))
        self.assertTrue(p.conditional_unique)
        # The outside-class truth 00001000 still agrees with that one label.
        self.assertEqual(np.array([0, 0, 0, 0, 1, 0, 0, 0])[0], 0)

    def test_observations_not_mutated(self):
        s = BehaviorSpace(8, 4, 1)
        x = np.array([-1, 0, -1, -1, 1, -1, -1, -1])
        before = x.copy()
        s.posterior(x)
        s.decode(x)
        np.testing.assert_array_equal(before, x)

    def test_invalid_inputs_rejected(self):
        for args in ((1,0,0), (8,9,0), (8,4,-1), (8,4,float("nan"))):
            with self.assertRaises(ValueError):
                BehaviorSpace(*args)
        with self.assertRaises(ValueError):
            BehaviorSpace(8,4).posterior(np.zeros(9))
        with self.assertRaises(ValueError):
            BehaviorSpace(8,4).posterior(np.full(8,2))

    def test_costs_validated(self):
        p = BehaviorSpace(8,4).posterior(np.full(8,-1))
        for costs in (np.zeros(8), np.ones(9), np.full(8, float("nan"))):
            with self.assertRaises(ValueError):
                p.choose_query(costs)

    def test_fit_uses_training_moment(self):
        rows = np.array(list(itertools.product((0,1), repeat=8)),np.uint8)
        rows = rows[np.count_nonzero(rows!=np.roll(rows,1,axis=1),axis=1)<=2]
        self.assertAlmostEqual(fit_penalty(rows,2),0)
        constant = np.zeros((2,8), np.uint8)
        self.assertGreater(fit_penalty(constant,4),7.9)
        with self.assertRaises(ValueError):
            fit_penalty(np.array([[0,1,0,1]]),2)


class EncoderTests(unittest.TestCase):
    def test_validation(self):
        for table in (bytes(255), bytes([2])*256, [0]*256):
            with self.assertRaises(ValueError):
                emit_scan(table,"count")
        with self.assertRaises(ValueError):
            emit_scan(bytes(256),"unknown")

    def test_elf_layout(self):
        for mode in ("count","first","prefix"):
            elf = emit_demo_elf(bytes(256),mode,b"abc")
            self.assertEqual(elf[:4],b"\x7fELF")
            self.assertEqual(struct.unpack_from("<Q",elf,96)[0],len(elf))
            self.assertEqual(struct.unpack_from("<Q",elf,24)[0],0x400000+120)

    def test_encoder_deterministic(self):
        table=bytes(i%2 for i in range(256))
        self.assertEqual(emit_scan(table,"count"),emit_scan(table,"count"))
        self.assertNotEqual(emit_scan(table,"first"),emit_scan(table,"prefix"))


class NativeLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The repository's shared checker requires GCC; fail, do not silently skip.
        if not shutil.which("gcc"):
            raise RuntimeError("GCC required for native behavior-loop tests")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        source = Path(__file__).resolve().parents[1]/"tools/behavior_native_runner.c"
        cls.runner = cls.root/"runner"
        subprocess.run(["gcc","-O2","-std=c11","-Wall","-Wextra","-Werror",
                        str(source),"-o",str(cls.runner)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_code(self, code, cases):
        (self.root/"fn.bin").write_bytes(code)
        (self.root/"cases.bin").write_bytes(struct.pack("<I",len(cases)) +
            b"".join(struct.pack("<I",len(c))+c for c in cases))
        return subprocess.run([str(self.runner),str(self.root/"fn.bin"),str(self.root/"cases.bin")],
                              capture_output=True,timeout=3)

    def test_all_single_bytes_and_long_boundaries(self):
        table=bytes(int(x%3==1) for x in range(256))
        cases=[b""]+[bytes([x]) for x in range(256)]
        cases += [b"\x01"*131072, b"\x00"*131071+b"\x01", b"\x01"*131071+b"\x00"]
        for mode in ("count","first","prefix"):
            result=self.run_code(emit_scan(table,mode),cases)
            self.assertEqual(result.returncode,0,result.stderr)
            expected=[]
            for c in cases:
                bits=[table[x] for x in c]
                if mode=="count":
                    expected.append(sum(bits))
                else:
                    stop=1 if mode=="first" else 0
                    expected.append(next((i for i,b in enumerate(bits) if b==stop),len(c)))
            self.assertEqual(list(struct.unpack("<"+"Q"*len(cases),result.stdout)),expected)

    def test_forbidden_syscall_rejected(self):
        # getpid is intentionally absent from the whitelist.
        r=self.run_code(bytes.fromhex("b8270000000f05c3"),[b""])
        self.assertNotEqual(r.returncode,0)

    def test_out_of_bounds_read_hits_guard(self):
        r=self.run_code(bytes.fromhex("0fb607c3"),[b""])
        self.assertNotEqual(r.returncode,0)

    def test_write_to_input_rejected(self):
        r=self.run_code(bytes.fromhex("c60700c3"),[b"A"])
        self.assertNotEqual(r.returncode,0)


if __name__=="__main__":
    unittest.main()
