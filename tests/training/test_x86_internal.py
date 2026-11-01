from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import torch

from lightnest.architecture.x86_internal import EXITED, FAULT, UNSUPPORTED, X86Internal
from lightnest.runtime.execute import Case, ExecutableRunner
from lightnest.runtime.internal_elf import load_static_elf


def build(root, source, name):
    assembly = root / (name + '.s'); obj = root / (name + '.o'); image = root / name
    assembly.write_text('.global _start\n.text\n_start:\n' + source + '\n.section .note.GNU-stack,"",@progbits\n')
    subprocess.run(['as', '--64', '-o', str(obj), str(assembly)], check=True, capture_output=True)
    subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(image), str(obj)], check=True, capture_output=True)
    return image.read_bytes()


class X86InternalTests(unittest.TestCase):
    def test_real_elf_matches_native_cpu_and_cuda(self):
        sources = {
            'hello': ('mov $1,%edi\nlea msg(%rip),%rsi\nmov $8,%edx\nmov $1,%eax\nsyscall\n'
                      'xor %edi,%edi\nmov $60,%eax\nsyscall\n.section .rodata\nmsg: .ascii "RGB.OK!\\n"'),
            'exit37': 'mov $37,%edi\nmov $60,%eax\nsyscall',
            'conditional': ('mov $0xffffffff,%eax\nadd $1,%eax\njz yes\nmov $5,%edi\njmp done\n'
                            'yes: mov $9,%edi\ndone: mov $60,%eax\nsyscall'),
            'stderr': ('mov $2,%edi\nlea msg(%rip),%rsi\nmov $3,%edx\nmov $1,%eax\nsyscall\n'
                       'mov $7,%edi\nmov $60,%eax\nsyscall\n.section .rodata\nmsg: .ascii "bad"'),
        }
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-test-') as temp:
            root = Path(temp); runner = ExecutableRunner(root / 'native', cache=False)
            for name, source in sources.items():
                raw = build(root, source, name)
                native = runner.run(raw, Case())
                self.assertEqual(native['status'], 'ok')
                for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
                    with self.subTest(name=name, device=device):
                        x = load_static_elf(raw, device=device)
                        result = X86Internal().to(device)(x['memory'], x['readable'], x['executable'], x['entry'], steps=32)
                        self.assertEqual(result['status'].item(), EXITED)
                        self.assertEqual(result['exit_code'].item(), native['returncode'])
                        for fd, key in enumerate(('stdout', 'stderr')):
                            size = result['output_lengths'][0,fd].item()
                            self.assertEqual(bytes(result['output'][0,fd,:size].tolist()).hex(), native[key])

    def test_batched_random_elf_outputs_against_native(self):
        rng = random.Random(20261101)
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-batch-') as temp:
            root = Path(temp); runner = ExecutableRunner(root / 'native-batch', cache=False)
            images, expected = [], []
            for i in range(32):
                payload = bytes(rng.randrange(256) for _ in range(8)); status = rng.randrange(256)
                source = ('mov $1,%edi\nlea msg(%rip),%rsi\nmov $8,%edx\nmov $1,%eax\nsyscall\n'
                          f'mov ${status},%edi\nmov $60,%eax\nsyscall\n.section .rodata\nmsg: .byte '
                          + ','.join(str(value) for value in payload))
                raw = build(root, source, f'case-{i}')
                native = runner.run(raw, Case())
                self.assertEqual(native['status'], 'ok')
                self.assertEqual(bytes.fromhex(native['stdout']), payload)
                self.assertEqual(native['returncode'], status)
                images.append(raw); expected.append((payload, status))
            from lightnest.architecture.elf_tensor_loader import ELFTensorLoader
            capacity = max(map(len, images))
            for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
                data = torch.zeros((len(images),capacity), dtype=torch.long, device=device)
                for i, image in enumerate(images): data[i,:len(image)] = torch.tensor(list(image), device=device)
                lengths = torch.tensor([len(image) for image in images], device=device)
                loaded = ELFTensorLoader().to(device)(data, lengths)
                result = X86Internal().to(device)(loaded['memory'], loaded['readable'], loaded['executable'], loaded['entry'], steps=16)
                self.assertTrue(loaded['valid'].all().item())
                self.assertTrue(result['status'].eq(EXITED).all().item())
                for i, (payload, status) in enumerate(expected):
                    self.assertEqual(result['exit_code'][i].item(), status)
                    self.assertEqual(bytes(result['output'][i,0,:8].tolist()), payload)

    def test_unsupported_opcode_and_output_overflow_are_not_success(self):
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-test-') as temp:
            root = Path(temp)
            bad = build(root, 'ud2', 'unsupported')
            x = load_static_elf(bad)
            result = X86Internal()(x['memory'], x['readable'], x['executable'], x['entry'], steps=2)
            self.assertEqual(result['status'].item(), UNSUPPORTED)
            raw = build(root, 'mov $1,%edi\nlea msg(%rip),%rsi\nmov $8,%edx\nmov $1,%eax\nsyscall\n'
                        'mov $60,%eax\nsyscall\n.section .rodata\nmsg: .ascii "12345678"', 'overflow')
            x = load_static_elf(raw)
            result = X86Internal(output_size=4)(x['memory'], x['readable'], x['executable'], x['entry'], steps=10)
            self.assertEqual(result['status'].item(), FAULT)
            self.assertEqual(result['output_lengths'].sum().item(), 0)

    def test_real_openat_instruction_uses_internal_learned_environment(self):
        class MissingEnvironment(torch.nn.Module):
            path_size = 64
            def forward(self, path, flags):
                return torch.tensor([-10., 10., -10.], device=path.device).expand(path.shape[0], -1)

        source = ('mov $257,%eax\nmov $0,%edi\nlea path(%rip),%rsi\nmov $0,%edx\nsyscall\n'
                  'mov %eax,%edi\nmov $60,%eax\nsyscall\n.section .rodata\npath: .asciz "/nonexistent"')
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-test-') as temp:
            raw = build(Path(temp), source, 'openat')
            for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
                x = load_static_elf(raw, device=device)
                unsupported = X86Internal().to(device)(x['memory'], x['readable'], x['executable'], x['entry'], steps=12)
                self.assertEqual(unsupported['status'].item(), UNSUPPORTED)
                model = X86Internal(environment=MissingEnvironment()).to(device)
                result = model(x['memory'], x['readable'], x['executable'], x['entry'], steps=12)
                self.assertEqual(result['status'].item(), EXITED)
                self.assertEqual(result['exit_code'].item(), 254)  # -ENOENT as process status
                self.assertEqual(result['open_mask'].sum().item(), 1)

    def test_uninitialized_register_read_faults(self):
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-test-') as temp:
            raw = build(Path(temp), 'mov %ebx,%edi\nmov $60,%eax\nsyscall', 'undefined-register')
            for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
                x = load_static_elf(raw, device=device)
                result = X86Internal().to(device)(x['memory'], x['readable'], x['executable'], x['entry'], steps=8)
                self.assertEqual(result['status'].item(), FAULT)

    def test_branch_with_undefined_flags_faults(self):
        with tempfile.TemporaryDirectory(prefix='lightnest-x86-test-') as temp:
            raw = build(Path(temp), 'jz done\ndone: mov $60,%eax\nsyscall', 'undefined')
            x = load_static_elf(raw)
            result = X86Internal()(x['memory'], x['readable'], x['executable'], x['entry'], steps=3)
            self.assertEqual(result['status'].item(), FAULT)


if __name__ == '__main__': unittest.main()
