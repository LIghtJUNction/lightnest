"""Observable ISA, execution-boundary and corruption checks for the P1 core."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lightnest.corpus import compile_generation_target
from archive.research.exact_core import FixedWordAdder
from archive.research.exact_x86 import ExactX86Runner
from lightnest.execute import Case, ExecutableRunner, same_result
from lightnest.iterate import Example, RGBEditor
from tools.probe_exact_x86 import build_program


def executable(body):
    assembly = ('.global _start\n.text\n_start:\n' + body + '\n'
                'mov %rax,output(%rip)\n mov $1,%eax\n mov $1,%edi\n'
                'lea output(%rip),%rsi\n mov $8,%edx\n syscall\n'
                'xor %edi,%edi\n mov $60,%eax\n syscall\n'
                '.bss\noutput: .skip 8\n.section .note.GNU-stack,"",@progbits\n')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'a.s').write_text(assembly)
        subprocess.run(['as', '--64', '-o', str(root / 'a.o'), str(root / 'a.s')], check=True, capture_output=True)
        subprocess.run(['ld', '--build-id=none', '-o', str(root / 'a'), str(root / 'a.o')], check=True, capture_output=True)
        return (root / 'a').read_bytes()


class ExactX86Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def compare(self, raw, case=Case()):
        with tempfile.TemporaryDirectory() as temporary:
            native = ExecutableRunner(Path(temporary)).run(raw, case)
        self.assertEqual(native['status'], 'ok', bytes.fromhex(native['stderr']))
        results = []
        for dtype in (torch.float32, torch.bfloat16):
            result = ExactX86Runner(dtype=dtype).run(raw, case)
            self.assertTrue(same_result(result, native), result)
            self.assertEqual(result['evidence'], 'compiled_core')
            self.assertEqual(result['native_calls'], 0)
            self.assertEqual(result['reference_corrections'], 0)
            results.append(result)
        self.assertEqual(results[0]['memory_sha256'], results[1]['memory_sha256'])
        return results[0]

    def test_existing_complete_elf_generation_curriculum(self):
        for spec in ({'exit': 37}, {'stdout': b'RGB.OK!\n'.hex()}):
            with self.subTest(spec=spec): self.compare(compile_generation_target(spec))

    def test_carry_borrow_and_signed_overflow_at_64_bits(self):
        for body in ('mov $-1,%rax\n add $1,%rax\n setb %al',
                     'mov $0,%rax\n sub $1,%rax\n setb %al',
                     'mov $0x7fffffffffffffff,%rax\n add $1,%rax\n seto %al',
                     'mov $0x8000000000000000,%rax\n sub $1,%rax\n seto %al'):
            with self.subTest(body=body): self.compare(executable(body))

    def test_partial_register_writes_and_zero_extension(self):
        for body in ('mov $-1,%rax\n mov $3,%al\n mov $7,%ah',
                     'mov $-1,%rax\n mov $0x1234,%ax',
                     'mov $-1,%rax\n mov $0x80000001,%eax'):
            with self.subTest(body=body): self.compare(executable(body))

    def test_false_conditional_move_clears_the_upper_half(self):
        self.compare(executable('mov $-1,%rax\n mov $7,%edx\n xor %ecx,%ecx\n cmovne %edx,%eax'))

    def test_undefined_shift_flag_cannot_choose_a_branch(self):
        result = ExactX86Runner().run(executable('mov $1,%eax\n shl $8,%al\n setb %al'))
        self.assertEqual(result['status'], 'undefined_input')

    def test_signed_unsigned_and_parity_branch_conditions(self):
        for jump in ('ja', 'jb', 'jge', 'jl', 'jle', 'jg', 'jz', 'jnz', 'jp', 'jnp'):
            body = f'mov $0x80000000,%eax\n cmp $1,%eax\n {jump} yes\n mov $7,%eax\n jmp done\nyes: mov $11,%eax\ndone:'
            with self.subTest(jump=jump): self.compare(executable(body))

    def test_variable_shifts_and_zero_count_flags(self):
        for shift in ('shl', 'shr', 'sar'):
            for count in (0, 1, 7, 31, 32, 63):
                body = f'mov $0x8123456789abcdef,%rax\n mov ${count},%ecx\n {shift} %cl,%rax'
                with self.subTest(shift=shift, count=count): self.compare(executable(body))

    def test_call_stack_and_memory_roundtrip(self):
        self.compare(executable('xor %ebp,%ebp\n mov $19,%eax\n call f\n jmp done\nf: push %rbp\n mov %rsp,%rbp\n add $23,%eax\n leave\n ret\ndone:'))

    def test_compiled_c_with_stdin_branches_loops_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for family in ('arithmetic32', 'arithmetic64', 'unsigned_select', 'signed_select', 'loop', 'memory', 'call'):
                with self.subTest(family=family):
                    program = build_program(Path(temporary) / family, family)
                    size = program['width'] // 8
                    stdin = ((1 << program['width']) - 1).to_bytes(size, 'little') + (3).to_bytes(size, 'little')
                    self.compare(Path(program['path']).read_bytes(), Case(stdin=stdin))

    def test_uninitialized_register_and_stack_are_not_zero_filled_answers(self):
        for body in ('mov %rbx,%rax', 'mov -8(%rsp),%rax', 'setb %al'):
            result = ExactX86Runner().run(executable(body))
            self.assertEqual(result['status'], 'undefined_input', result)

    def test_fault_unsupported_and_budget_never_report_success(self):
        for body, status in (('mov $1,%rax\n mov %rax,_start(%rip)', 'memory_fault'),
                             ('mov $0,%rax\n mov (%rax),%rax', 'memory_fault'),
                             ('mov $39,%eax\n syscall', 'unsupported'),
                             ('jmp _start', 'budget')):
            result = ExactX86Runner(max_steps=16).run(executable(body))
            self.assertEqual(result['status'], status, result)
            self.assertIsNone(result['returncode'])

    def test_malformed_dynamic_and_argument_inputs_are_explicit(self):
        self.assertEqual(ExactX86Runner().run(b'bad')['status'], 'unsupported')
        self.assertEqual(ExactX86Runner().run(bytes(64))['status'], 'invalid_artifact')
        self.assertEqual(ExactX86Runner().run(Path('/usr/bin/true').read_bytes())['status'], 'unsupported')
        result = ExactX86Runner().run(compile_generation_target({'exit': 0}), Case(('x',)))
        self.assertEqual(result['status'], 'unsupported')

    def test_excess_segment_table_is_rejected_before_loading(self):
        raw = bytearray(compile_generation_target({'exit': 0}))
        raw[56:58] = (129).to_bytes(2, 'little')
        result = ExactX86Runner().run(bytes(raw))
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('table capacity', result['reason'])

    def test_state_is_rebuilt_and_bound_to_image_and_input(self):
        runner = ExactX86Runner()
        a = runner.run(compile_generation_target({'exit': 37}))
        b = runner.run(compile_generation_target({'exit': 91}))
        c = runner.run(compile_generation_target({'exit': 37}), Case(stdin=b'x'))
        self.assertEqual((a['returncode'], b['returncode'], c['returncode']), (37, 91, 37))
        self.assertEqual(len({r['state_binding_sha256'] for r in (a, b, c)}), 3)

    def test_existing_editor_can_validate_a_fixture_proposal_with_the_core(self):
        original = compile_generation_target({'exit': 37})
        offset = original.index(bytes.fromhex('bf25000000')) + 1
        damaged = bytearray(original); damaged[offset] = 91; damaged = bytes(damaged)

        class FixturePolicy:
            training = False
            recurrent_steps = 0
            green = SimpleNamespace(encoded_chunks=0, memory=SimpleNamespace(bank_builds=0))

            def thoughts(self, image, command, frames, **kwargs):
                self.recurrent_steps += 1; self.green.memory.bank_builds += 1
                logits = torch.full((len(image),), -20.); logits[offset] = 20.
                bits = torch.zeros(len(image), 8)
                bits[offset] = torch.tensor([20. if 37 >> bit & 1 else -20. for bit in range(8)])
                yield {'logits': logits, 'bits': bits, 'risk_logit': torch.tensor(-20.),
                       'depth': 1, 'origin': 'blue'}, {'observation': SimpleNamespace(instances=1, payloads=[b'fixture'])}

        runner = ExactX86Runner()
        expected = {key: value for key, value in runner.run(original).items()
                    if key in ('status', 'returncode', 'stdout', 'stderr')}
        result = RGBEditor(FixturePolicy(), runner).repair(damaged, 'restore visible behavior', [Example(Case(), expected)])
        self.assertTrue(result['stop']); self.assertEqual(result['bytes'], original)
        self.assertEqual(result['edits'], 1)

    def test_corruption_is_detected_without_native_fallback(self):
        raw = executable('mov $19,%eax\n add $23,%eax')
        with tempfile.TemporaryDirectory() as temporary:
            native = ExecutableRunner(Path(temporary)).run(raw, Case())
        original = FixedWordAdder.forward

        def corrupt(module, a, b, carry_in=None):
            value, carry = original(module, a, b, carry_in)
            value = value.clone(); value[:, 0] = 1 - value[:, 0]
            return value, carry

        with patch.object(FixedWordAdder, 'forward', corrupt), patch('subprocess.run', side_effect=AssertionError('native fallback')):
            result = ExactX86Runner().run(raw)
        self.assertFalse(same_result(result, native))
        self.assertEqual(result['native_calls'], 0)


if __name__ == '__main__':
    unittest.main()
