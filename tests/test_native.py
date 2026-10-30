from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from lightnest.isa import assemble,canonical_arch,decode,elf_functions,host_architecture,llvm_targets
from archive.universal_native.sandbox import Sandbox
from archive.universal_native.model import Batch,Config,MachineNetwork,load,loss_for,save,validate
from archive.universal_native.corpus import compile_c,reference,specification,translate_rv32
from archive.universal_native.repair import VerifiedRepair


def row(code='4889f84801f0c3'):
    return {'architecture':'x86_64','command':'replace addition with subtraction','binary':code,
            'examples':[{'inputs':[5,2,0],'output':3},{'inputs':[7,1,0],'output':6}]}


class DecoderTests(unittest.TestCase):
    def test_host_and_inventory(self):
        self.assertEqual(canonical_arch('AMD64'),'x86_64')
        for name in ('x86_64','aarch64','riscv32','riscv64','arm','ppc64le','mipsel'):
            self.assertIn(name,llvm_targets())

    def test_x86_instruction_families_decode_without_execution(self):
        examples={'integer':'4889f84801f0c3','sse':'0f58c1','avx2':'c5fdfec1',
                  'avx512':'62f17d48fec1','aes':'660f38dcc1','x87':'d9fe',
                  'popcnt':'f3480fb8c1','atomic':'f0480fc118','system':'0f050f32','virtualization':'0f01c2'}
        for family,code in examples.items():
            with self.subTest(family=family):
                raw=bytes.fromhex(code);instructions=decode('x86_64',raw)
                self.assertEqual(b''.join(i.raw for i in instructions),raw)
                self.assertEqual([i.offset for i in instructions],sorted(i.offset for i in instructions))

    def test_foreign_isa_round_trip_is_not_execution(self):
        examples={'aarch64':'c0035fd6','arm':'1eff2fe1','thumb':'7047','riscv32':'67800000',
                  'riscv64':'67800000','ppc64le':'2000804e','mipsel':'0800e00300000000',
                  'systemz':'07fe','loongarch64':'2000004c'}
        for architecture,code in examples.items():
            with self.subTest(architecture=architecture):
                self.assertEqual(b''.join(i.raw for i in decode(architecture,bytes.fromhex(code))),bytes.fromhex(code))

    def test_truncated_and_unknown_targets_fail(self):
        for architecture,code in [('x86_64',b'\x0f'),('riscv64',b'\x67'),('no-such-cpu',b'1234')]:
            with self.subTest(architecture=architecture),self.assertRaises(ValueError):decode(architecture,code)

    def test_multitarget_assembler_and_unresolved_relocations(self):
        for architecture,source in (('x86_64','addq %rbx,%rax\nretq\n'),('aarch64','add x0,x0,x1\nret\n'),('riscv64','add a0,a0,a1\nret\n')):
            with self.subTest(architecture=architecture):
                code=assemble(architecture,source)
                self.assertEqual(b''.join(i.raw for i in decode(architecture,code)),code)
        with self.assertRaises(ValueError):assemble('x86_64','call missing_function\n')

    def test_elf_function_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'function.o'
            subprocess.run(['cc','-x','c','-O2','-c','-o',str(path),'-'],input=b'unsigned long f(unsigned long a){return a+1;}',check=True)
            functions=elf_functions(path.read_bytes())
            self.assertEqual(functions[0]['name'],'f');self.assertEqual(functions[0]['architecture'],'x86_64')
            self.assertTrue(functions[0]['instructions'])


class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(17);return MachineNetwork(Config(width=32,rounds=2))

    def test_input_rejects_labels_and_bad_example(self):
        with self.assertRaises(ValueError):validate(row()|{'target':'c3'},Config())
        for example in ({'inputs':[1,2],'output':0},{'inputs':[1,2,3],'output':-1},{'inputs':[True,2,3],'output':0}):
            with self.subTest(example=example),self.assertRaises(ValueError):validate(row()|{'examples':[example]},Config())

    def test_invalid_config(self):
        for kwargs in ({'rounds':0},{'width':33},{'max_bytes':65536},{'max_instructions':2048}):
            with self.assertRaises(ValueError):Config(**kwargs)

    def test_variable_instruction_lengths_preserve_byte_positions(self):
        batch=Batch.encode([row()],Config())
        offsets=batch.offsets[batch.raw.ne(0)].tolist()
        self.assertEqual(offsets,list(range(7)))
        self.assertEqual(batch.valid.sum().item(),3)

    def test_padding_batch_invariance_and_finite_gradients(self):
        model=self.model();a=row();b=row('4889f84801f04883c001c3')
        batch=Batch.encode([a,b],model.config);z,bits=model(batch)
        loss=loss_for(z,bits,batch,[a|{'target':'4889f84829f0c3'},b|{'target':b['binary']}]);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            single=model(Batch.encode([a],model.config))
            together=model(Batch.encode([a,b],model.config))
        torch.testing.assert_close(single[0][0],together[0][0,:7],rtol=1e-4,atol=1e-5)
        self.assertTrue(torch.isneginf(together[0][0,7:]).all())

    def test_examples_change_network_conditioning(self):
        model=self.model().eval();a=row();b=row();b['examples']=[{'inputs':[5,2,0],'output':123456}]
        with torch.no_grad():z=model(Batch.encode([a,b],model.config))[0]
        self.assertFalse(torch.equal(z[0],z[1]))

    def test_checkpoint_round_trip(self):
        model=self.model().eval()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';save(path,model,{'fixture':True})
            self.assertEqual(model.predict([row()]),load(path).predict([row()]))
            with self.assertRaises(FileExistsError):save(path,model,{})

    def test_untrained_platform_refused(self):
        with self.assertRaises(ValueError):
            self.model().eval().predict([row()|{'architecture':'aarch64','binary':'c0035fd6'}])

    def test_all_negative_is_stop_and_budget_is_rejected(self):
        model=self.model().eval()
        with torch.no_grad():model.edit.weight.zero_();model.edit.bias.fill_(-20)
        self.assertTrue(model.predict([row()])[0]['stop'])
        long=row('90'*96+'c3')
        with torch.no_grad():model.edit.bias.fill_(20)
        pred=model.predict([long])[0]
        self.assertEqual(pred['status'],'budget');self.assertFalse(pred['stop']);self.assertEqual(pred['final'],long['binary'])


@unittest.skipUnless(host_architecture()=='x86_64','native ABI currently x86-64')
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='lightnest-sandbox-test-');self.addCleanup(self.temp.cleanup)
        self.vm=Sandbox(Path(self.temp.name))

    def test_native_arithmetic(self):
        self.assertEqual(self.vm.run(bytes.fromhex('4889f84801f0c3'),[[5,2,0],[7,3,0]])['outputs'],[7,10])

    def test_disallowed_syscall_and_privileged_instruction(self):
        for code in ('b8020000000f05c3','0f32c3'):
            self.assertEqual(self.vm.run(bytes.fromhex(code),[[0,0,0]])['status'],'rejected')

    def test_large_write_cannot_bypass_high_word_check(self):
        code=bytes.fromhex('b801000000bf0100000048ba01000000010000000f05c3')
        self.assertEqual(self.vm.run(code,[[0,0,0]])['status'],'rejected')

    def test_loop_is_bounded(self):
        self.assertNotEqual(self.vm.run(bytes.fromhex('ebfe'),[[0,0,0]],timeout=1.5)['status'],'ok')

    def test_compiler_against_independent_reference(self):
        inputs=[[1,2,3],[2**64-1,7,1],[7,7,0]]
        for family in ('add_sub','xor_or','and_or','left_right','rotate','minimum_maximum','constant','compare'):
            with self.subTest(family=family):
                source,*_=specification(family,718,True);code=compile_c(source)
                self.assertEqual(self.vm.run(code,inputs)['outputs'],[reference(family,718,x) for x in inputs])

    def test_historical_add_translation_has_same_result(self):
        raw=bytes.fromhex('83200000032140003385200073001000')
        code=translate_rv32(raw)
        self.assertEqual(self.vm.run(code,[[8,3,0]])['outputs'],[11])

    def proposal(self):
        class Proposal(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.config=Config()
            def forward(self,batch):
                raw=batch.codes[0];z=torch.full((1,len(raw)),-10.);z[0,4]=10.
                values=list(raw);values[4]=0x21
                bits=torch.tensor([[[10. if v>>k&1 else -10. for k in range(8)] for v in values]])
                bits[0,4,3]=-.01
                return z,bits
        return Proposal().eval()

    def test_verified_decoder_corrects_uncertain_bit_using_visible_cases(self):
        repair=VerifiedRepair(self.proposal(),self.vm,candidates=16)
        pred=repair.repair(row())
        self.assertEqual(pred['final'],'4889f84829f0c3')
        self.assertEqual(pred['status'],'sample_checked_candidate')
        self.assertGreaterEqual(pred['native_checks'],2)

    def test_verified_decoder_preserves_observed_correct_input(self):
        pred=VerifiedRepair(self.proposal(),self.vm).repair(row('4889f84829f0c3'))
        self.assertEqual(pred['status'],'observed_unchanged');self.assertEqual(pred['patches'],[])
        self.assertEqual(pred['candidate_attempts'],1)

    def test_verified_decoder_budget_and_label_boundary(self):
        repair=VerifiedRepair(self.proposal(),self.vm,candidates=1)
        pred=repair.repair(row());self.assertFalse(pred['stop']);self.assertEqual(pred['final'],row()['binary'])
        with self.assertRaises(ValueError):repair.repair(row()|{'target':'4889f84829f0c3'})


if __name__=='__main__':unittest.main()
