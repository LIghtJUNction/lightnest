from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lightnest.architecture.x86_world import X86World
from lightnest.runtime.execute import Case, ExecutableRunner
from tests.training.test_x86_internal import build


class X86WorldTests(unittest.TestCase):
    def run_device(self, device):
        source = ('mov $1,%edi\nlea msg(%rip),%rsi\nmov $3,%edx\nmov $1,%eax\nsyscall\n'
                  'xor %edi,%edi\nmov $60,%eax\nsyscall\n.section .rodata\nmsg: .ascii "Hi\\n"')
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-world-') as temp:
            root = Path(temp); good = build(root, source, 'hello')
            bad = bytearray(good); offset = bad.index(b'Hi\n') + 1; bad[offset] = ord('o')
            capacity = max(768, len(good))
            model = X86World(condition_width=1, candidates=1, image_capacity=capacity).to(device).eval()
            with torch.no_grad():
                model.proposal.weight.zero_(); model.proposal.bias.fill_(-10)
                model.length.weight.zero_(); model.length.bias.fill_(-10)
                model.refine.weight.zero_(); model.refine.bias.zero_()
                model.refine.weight[0, 4] = 2
                biases = model.proposal.bias.view(1, capacity, 256)
                weights = model.proposal.weight.view(1, capacity, 256, 1)
                biases[0, torch.arange(len(bad), device=device), torch.tensor(list(bad), device=device)] = 10
                biases[0, len(bad):, 0] = 10
                biases[0, offset, ord('o')] = 1
                biases[0, offset, ord('i')] = -1
                weights[0, offset, ord('i'), 0] = 3
                model.length.bias[len(good)-1] = 10
                expected = torch.zeros((1,64), dtype=torch.long, device=device)
                expected[0,:3] = torch.tensor(list(b'Hi\n'), device=device)
                with patch('subprocess.run', side_effect=AssertionError('external process during model iteration')):
                    result = model(torch.zeros((1,1), device=device), expected,
                                   torch.tensor([3], device=device), torch.tensor([0], device=device), steps=16, rounds=2)
            self.assertEqual(result['round_passed'].tolist(), [[[False], [True]]])
            self.assertTrue(result['accepted'].item())
            self.assertEqual(result['selected_round'].item(), 1)
            generated = bytes(result['image'][0,:result['length'][0]].tolist())
            self.assertEqual(generated, good)
            with torch.no_grad():
                biases[0, offset, ord('i')] = 10
                biases[0, offset, ord('o')] = -10
                model.refine.bias.fill_(5)
                stable = model(torch.zeros((1,1), device=device), expected,
                               torch.tensor([3], device=device), torch.tensor([0], device=device), steps=16, rounds=2)
            self.assertEqual(stable['round_passed'].tolist(), [[[True], [True]]])
            native = ExecutableRunner(root / 'native').run(generated, Case())
            self.assertEqual(bytes.fromhex(native['stdout']), b'Hi\n')
            self.assertEqual(native['returncode'], 0)

    def test_two_round_training_backpropagates_through_feedback_controller(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            model = X86World(condition_width=2, candidates=1, image_capacity=64, memory_size=128, output_size=8).to(device)
            result = model(torch.zeros((1,2), device=device), torch.zeros((1,8), dtype=torch.long, device=device),
                           torch.tensor([0], device=device), torch.tensor([0], device=device), steps=2, rounds=2)
            loss = result['byte_logits'][:,1].square().mean() + result['length_logits'][:,1].square().mean()
            loss.backward()
            self.assertIsNotNone(model.refine.weight.grad)
            self.assertTrue(torch.isfinite(model.refine.weight.grad).all().item())
            self.assertGreater(model.refine.weight.grad.abs().sum().item(), 0)

    def test_cpu_internal_refinement(self): self.run_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_internal_refinement(self): self.run_device('cuda')


if __name__ == '__main__': unittest.main()
