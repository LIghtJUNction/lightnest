"""Mechanism checks which do not require external checkpoint or dataset files."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SourceOnlyTests(unittest.TestCase):
    def test_required_source(self):
        for name in ("hybrid/model.py",
                     "baseline/LightNest_RGB_v2/rgb/network.py",
                     "baseline/LightNest_RGB_v2/runtime/base/lightnest/work/breakthrough-20261018-0029/arch/breakthrough_model.py",
                     "tools/template_io/template_io.py"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_action_codec(self):
        from hybrid.model import encode_action, decode_action
        for p in (0, 17, 1023):
            for v in (0, 1, 127, 255):
                original = {"position": p, "value": v, "stop": False}
                self.assertEqual(decode_action(encode_action(original)), original)
        self.assertTrue(decode_action("STOP")["stop"])

    def test_reject_invalid_action(self):
        from hybrid.model import decode_action
        for value in ("PATCH -1 00", "PATCH 1024 10", "PATCH 1 gg", "THINK"):
            with self.assertRaises(ValueError):
                decode_action(value)

    def test_bounded_program(self):
        import hybrid  # adds current baseline import root
        from rgb.runtime import common
        code = common.assemble("lw x1, 0(x0)\nlw x2, 4(x0)\nadd x10, x1, x2\nebreak\n")
        self.assertEqual(len(code), 16)

if __name__ == "__main__":
    unittest.main()
