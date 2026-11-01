import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lightnest.training.training import train_generator


class GeneratorDeviceTests(unittest.TestCase):
    def _one_step(self, device):
        with tempfile.TemporaryDirectory(prefix='lightnest-generator-test-') as folder:
            out = Path(folder)
            for name in ('training', 'weights', 'generation', 'objects'):
                (out / name).mkdir()
            raw = b'\x7fELF' + bytes(range(32))
            (out / 'generation' / 'sample.elf').write_bytes(raw)
            (out / 'objects' / 'native.elf').write_bytes(raw)
            record = {'id': 'one', 'sha256': 'sample', 'spec': {'exit': 7}}
            for split in ('train', 'validation'):
                (out / 'generation' / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
            (out / 'executables.json').write_text(json.dumps([{'split': 'train', 'sha256': 'native'}]))
            (out / 'corpus.json').write_text('{}')
            with patch('lightnest.training.training.verify_corpus'), patch('lightnest.training.training.ExecutableRunner'):
                train_generator(out, 923457, steps=1, width=16, device=device)
            result = json.loads((out / 'training/generator-923457/complete.json').read_text())
            self.assertEqual(result['device'], device)
            self.assertEqual(result['updates'], 1)
            payload = torch.load(out / 'weights/generator-923457.pt', map_location='cpu', weights_only=True)
            self.assertEqual(payload['metadata']['device'], device)

    def test_one_cpu_step_records_device(self):
        self._one_step('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_one_cuda_step_records_device(self):
        self._one_step('cuda')

    def test_missing_cuda_fails_before_preparing_files(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                train_generator('/tmp/not-a-corpus', device='cuda')
