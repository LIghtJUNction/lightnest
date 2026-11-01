import random
import unittest
import torch
from lightnest.architecture.ptx_internal import PTXIntegerOps


class PTXIntegerTests(unittest.TestCase):
    def test_real_ptx_integer_semantics(self):
        rng = random.Random(20261101)
        values = [0, 1, 0xffffffff, 0x80000000] + [rng.randrange(1 << 32) for _ in range(1000)]
        other = list(reversed(values))
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            left = torch.tensor(values, dtype=torch.long, device=device)
            right = torch.tensor(other, dtype=torch.long, device=device)
            module = PTXIntegerOps().to(device)
            self.assertEqual(module(left, right, operation='add.u32').tolist(), [(a+b) & 0xffffffff for a,b in zip(values,other)])
            self.assertEqual(module(left, right, operation='mul.lo.u32').tolist(), [(a*b) & 0xffffffff for a,b in zip(values,other)])
            with self.assertRaises(ValueError): module(left,right,operation='unknown')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_against_physical_gpu_ptx(self):
        from tests.training.ptx_hardware import verify_hardware_ptx
        verify_hardware_ptx()


if __name__ == '__main__': unittest.main()
