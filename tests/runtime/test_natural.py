import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lightnest.runtime import natural


class NaturalTests(unittest.TestCase):
    def test_edit_preserves_source_and_checks_bytes(self):
        raw = Path('/usr/bin/true').read_bytes()
        # Restrict the test image to the API limit without relying on a system binary size.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.write_bytes(raw)
            output = Path(folder) / 'edited'
            with patch.object(natural, '_image', return_value=(raw, natural.inspect_elf(raw))):
                value = {'patches': [{'offset': 9, 'before': raw[9:10].hex(), 'after': bytes([raw[9] ^ 1]).hex()}]}
                with patch.object(natural, 'ask_model', return_value=json.dumps(value)):
                    result = natural.edit(source, 'change a byte', output)
                self.assertEqual(source.read_bytes(), raw)
                self.assertEqual(output.read_bytes()[9], raw[9] ^ 1)
                self.assertFalse(result['behavior_verified'])
                with patch.object(natural, 'ask_model', return_value=json.dumps(value)):
                    with self.assertRaises(ValueError): natural.edit(source, 'again', output)
                value['patches'][0]['before'] = 'ff'
                with patch.object(natural, 'ask_model', return_value=json.dumps(value)):
                    with self.assertRaises(ValueError): natural.edit(source, 'bad', Path(folder) / 'bad')

    def test_entry_disassembly_stays_within_executable_section(self):
        raw = bytes.fromhex('b8010000000f05') + b'RGB.OK!\n'
        info = {'machine': 'EM_X86_64', 'entry': 0x400000,
                'segments': [{'type': 'PT_LOAD', 'flags': 5, 'offset': 0, 'address': 0x400000, 'file_size': len(raw)}],
                'sections': [{'flags': 4, 'offset': 0, 'size': 7}]}
        text = natural._entry_disassembly(raw, info)
        self.assertIn('syscall', text)
        self.assertNotIn('0x400007:', text)

    def test_create_and_edit_execute_with_stubbed_model(self):
        from lightnest.runtime.execute import Case, ExecutableRunner

        assembly = ('.global _start\n.text\n_start:\n'
                    'mov $1,%eax\n mov $1,%edi\n lea message(%rip),%rsi\n'
                    'mov $3,%edx\n syscall\n mov $60,%eax\n xor %edi,%edi\n syscall\n'
                    'message: .ascii "Hi\\n"\n.section .note.GNU-stack,"",@progbits\n')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'fixture.s').write_text(assembly)
            subprocess.run(['as', '--64', '-o', str(root / 'fixture.o'), str(root / 'fixture.s')], check=True, capture_output=True)
            subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(root / 'fixture'), str(root / 'fixture.o')], check=True, capture_output=True)
            raw = (root / 'fixture').read_bytes()
            self.assertLessEqual(len(raw), natural.MAX_PROMPT_IMAGE)
            index = raw.index(b'Hi\n')
            reply = json.dumps({'hex': raw.hex()})
            with patch.object(natural, 'ask_model', side_effect=['{"hex":"00"}', reply]) as model:
                natural.create('print Hi', root / 'created')
                self.assertEqual(model.call_count, 2)
                self.assertIn('failed ELF validation', model.call_args.args[0])
            reply = json.dumps({'patches': [{'offset': index, 'before': '4869', 'after': '596f'}]})
            with patch.object(natural, 'ask_model', return_value=reply):
                natural.edit(root / 'created', 'change Hi to Yo', root / 'edited')
            runner = ExecutableRunner(root / 'runs')
            self.assertEqual(bytes.fromhex(runner.run((root / 'created').read_bytes(), Case())['stdout']), b'Hi\n')
            self.assertEqual(bytes.fromhex(runner.run((root / 'edited').read_bytes(), Case())['stdout']), b'Yo\n')

    def test_create_rejects_invalid_and_explain_records_no_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            with patch.object(natural, 'ask_model', return_value='{"hex":"00"}'):
                with self.assertRaises(ValueError): natural.create('hello', output)
            self.assertFalse(output.exists())
            with patch.object(natural, 'MAX_PROMPT_IMAGE', 1000000), patch.object(natural, 'ask_model', return_value='Likely exits successfully.'):
                result = natural.explain(Path('/usr/bin/true'))
            self.assertFalse(result['executed'])
            self.assertIn('exits', result['explanation'])
