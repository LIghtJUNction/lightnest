import importlib.util
import marshal
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from lightnest_rgb.packed import COOKIE, COOKIE_MAGIC, ENTRY, PackedArtifact, WRAPPER_MAGIC


def packed_image(code, *, expanded_override=None):
    raw = marshal.dumps(compile(code, '<fixture>', 'exec'))
    compressed = zlib.compress(raw) if expanded_override is None else zlib.compress(expanded_override)
    pyz = b'PYZ\0' + importlib.util.MAGIC_NUMBER + struct.pack('!i', 12 + len(compressed))
    pyz += compressed + marshal.dumps([('external.module', (0, 12, len(compressed)))])
    data = pyz + zlib.compress(raw)
    toc = bytearray()
    for name, offset, size, expanded, zipped, kind in [
        ('code.pyz', 0, len(pyz), len(pyz), 0, b'z'),
        ('entry', len(pyz), len(data) - len(pyz), len(raw), 1, b's'),
    ]:
        name = name.encode() + b'\0'
        length = ENTRY.size + len(name)
        toc.extend(ENTRY.pack(length, offset, size, expanded, zipped, kind) + name)
    cookie = COOKIE.pack(COOKIE_MAGIC, len(data) + len(toc) + COOKIE.size,
                         len(data), len(toc), 313, b'libpython3.13.so')
    return b'\x7fELF' + bytes(60) + data + toc + cookie


class PackedArtifactTests(unittest.TestCase):
    def test_both_code_containers_extract_byte_exact_code_without_running_it(self):
        source = 'def increase(value):\n    return value + 1\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'program'; path.write_bytes(packed_image(source))
            with PackedArtifact(path) as reader:
                module = reader.read_module('external.module')
                entry = reader.read_code_member('entry')
        self.assertEqual(module, entry)
        self.assertEqual(module[:4], importlib.util.MAGIC_NUMBER)
        expected = marshal.dumps(compile(source, '<fixture>', 'exec'))
        self.assertEqual(module[16:], expected)

    def test_large_wrapped_language_payload_is_not_read_or_materialized(self):
        base = packed_image('value = 1\n'); language_size = 64 << 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrapped'
            with path.open('wb') as file:
                file.write(b'\x7fELFstub' + base + b'GGUF')
                file.seek(8 + len(base) + language_size)
                file.write(WRAPPER_MAGIC + struct.pack('<QQ', len(base), language_size))
            with PackedArtifact(path) as reader:
                raw = reader.read_module('external.module')
                self.assertEqual(reader.wrapper['language_bytes'], language_size)
                self.assertLess(reader.bytes_read, 128 << 10)
                self.assertGreater(len(raw), 16)

    def test_expansion_and_invalid_wrapper_spans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oversize'
            path.write_bytes(packed_image('value = 1\n', expanded_override=b'x' * 65536))
            with PackedArtifact(path, max_member_bytes=1024) as reader:
                with self.assertRaises(ValueError):
                    reader.read_module('external.module')
            path.write_bytes(b'x' * 128 + WRAPPER_MAGIC + struct.pack('<QQ', 999999, 999999))
            with self.assertRaises(ValueError):
                PackedArtifact(path)


if __name__ == '__main__':
    unittest.main()
