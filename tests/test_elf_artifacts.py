from pathlib import Path
import tempfile
import unittest

from lightnest_rgb.artifact import BytePatch, apply_patches, digest, edit_file, inspect_elf, write_image
from lightnest_rgb.execute import Case, ExecutableRunner, same_result


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.raw = Path("/usr/bin/true").read_bytes()

    def test_real_executable_layout(self):
        info = inspect_elf(self.raw)
        self.assertEqual(info["machine"], "EM_X86_64")
        self.assertTrue(info["entry_in_executable_segment"])
        self.assertTrue(any(s["name"] == ".text" for s in info["sections"]))

    def test_transaction_preserves_source_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source", Path(temp) / "edited"
            source.write_bytes(self.raw); source.chmod(0o500)
            patch = BytePatch(9, self.raw[9:10], bytes([self.raw[9] ^ 1]))
            result = edit_file(source, destination, [patch], digest(self.raw))
            self.assertEqual(source.read_bytes(), self.raw)
            self.assertEqual(result["changed_bytes"], 1)
            self.assertEqual(destination.read_bytes(), apply_patches(self.raw, [patch], digest(self.raw)))
            with self.assertRaises(ValueError):
                edit_file(source, destination, [patch], digest(self.raw))
            with self.assertRaises(ValueError):
                apply_patches(self.raw, [patch], "0" * 64)

    def test_patch_precondition_and_overlap(self):
        good = BytePatch(9, self.raw[9:10], b"\x01")
        for patches in ([good, good], [BytePatch(len(self.raw), b"\0", b"\1")], [BytePatch(9, b"\xff", b"\0")]):
            with self.assertRaises(ValueError):
                apply_patches(self.raw, patches, digest(self.raw))

    def test_generation_writer_never_adds_missing_container(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "generated"
            with self.assertRaises(ValueError):
                write_image(path, b"\x90\xc3")
            self.assertFalse(path.exists())
            write_image(path, self.raw)
            self.assertEqual(path.read_bytes(), self.raw)
            with self.assertRaises(FileExistsError):
                write_image(path, self.raw)


class FullExecutableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lightnest-elf-test-")
        self.addCleanup(self.temp.cleanup)
        self.runner = ExecutableRunner(Path(self.temp.name))

    def test_real_sort_and_exact_cache(self):
        raw = Path("/usr/bin/sort").read_bytes(); case = Case((), b"z\na\na\n")
        first = self.runner.run(raw, case); repeated = self.runner.run(raw, case)
        self.assertEqual(first["returncode"], 0)
        self.assertEqual(bytes.fromhex(first["stdout"]), b"a\na\nz\n")
        self.assertTrue(repeated["cached"]); self.assertTrue(same_result(first, repeated))
        self.assertEqual(self.runner.native_calls, 1)

    def test_real_gzip_and_modified_image_cache_identity(self):
        raw = Path("/usr/bin/gzip").read_bytes(); case = Case(("-n", "-c"), b"repeat\n" * 4)
        first = self.runner.run(raw, case)
        changed = bytearray(raw); changed[9] ^= 1
        second = self.runner.run(bytes(changed), case)
        self.assertFalse(second["cached"])
        self.assertTrue(same_result(first, second))
        self.assertEqual(self.runner.native_calls, 2)

    def test_no_private_host_paths_are_mounted(self):
        # /home is deliberately absent from the child filesystem, even though
        # the real host has one. This executes an existing complete ELF.
        result = self.runner.run(Path("/usr/bin/ls").read_bytes(), Case(("/home",), b""))
        self.assertNotEqual(result["returncode"], 0)

    def test_expected_outputs_cannot_enter_runner_input(self):
        with self.assertRaises(ValueError):
            Case.parse({"args": [], "stdin": "", "expected": "answer"})


if __name__ == "__main__":
    unittest.main()
