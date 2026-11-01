from pathlib import Path
import tempfile
import unittest

import torch

from lightnest.architecture.elf_tensor_loader import ELFTensorLoader
from lightnest.architecture.x86_internal import EXITED, X86Internal
from lightnest.runtime.internal_elf import load_static_elf
from tests.training.test_x86_internal import build


class TensorELFLoaderTests(unittest.TestCase):
    def test_real_elf_matches_independent_loader_and_runs(self):
        source = ('mov $1,%edi\nlea msg(%rip),%rsi\nmov $3,%edx\nmov $1,%eax\nsyscall\n'
                  'xor %edi,%edi\nmov $60,%eax\nsyscall\n.section .rodata\nmsg: .ascii "Hi\\n"')
        with tempfile.TemporaryDirectory(prefix='lightnest-tensor-elf-') as temp:
            raw = build(Path(temp), source, 'hello')
            for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
                with self.subTest(device=device):
                    images = torch.tensor([list(raw)], dtype=torch.long, device=device)
                    result = ELFTensorLoader().to(device)(images, torch.tensor([len(raw)], device=device))
                    reference = load_static_elf(raw, device=device)
                    self.assertTrue(result['valid'].item())
                    for key in ('memory','readable','executable','entry'):
                        self.assertTrue(torch.equal(result[key], reference[key]), key)
                    execution = X86Internal().to(device)(result['memory'], result['readable'], result['executable'], result['entry'], steps=16)
                    self.assertEqual(execution['status'].item(), EXITED)
                    self.assertEqual(bytes(execution['output'][0,0,:3].tolist()), b'Hi\n')

    def test_invalid_header_and_truncated_image_are_rejected_in_batch(self):
        with tempfile.TemporaryDirectory(prefix='lightnest-tensor-elf-') as temp:
            raw = build(Path(temp), 'mov $60,%eax\nxor %edi,%edi\nsyscall', 'exit')
            valid = list(raw); invalid = valid.copy(); invalid[0] = 0
            images = torch.tensor([valid, invalid, valid], dtype=torch.long)
            result = ELFTensorLoader()(images, torch.tensor([len(raw), len(raw), 64]))
            self.assertEqual(result['valid'].tolist(), [True, False, False])


if __name__ == '__main__': unittest.main()
