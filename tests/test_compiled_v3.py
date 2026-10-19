"""Source-only safety, ISA, and compiler tests. Generated files stay in temp dirs."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from experiments.compiled_v3 import corpus, inputs, search, vm


def words(*instructions):return b''.join(struct.pack('<I',w) for w in instructions)
def r_word(f,h=0):return (h<<25)|(6<<20)|(5<<15)|(f<<12)|(10<<7)|0x33
LOAD_A=0x00052283
LOAD_B=0x00452303
RETURN=0x00008067


class CompiledV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('gcc'):raise unittest.SkipTest('gcc required for bounded interpreter')
        cls.temp=tempfile.TemporaryDirectory(prefix='lightnest-compiled-v3-')
        cls.root=Path(cls.temp.name);cls.machine=vm.NativeVM(cls.root/'runtime')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_input_contract(self):
        for a,n,k in (([0]*31,1,0),([0]*32,33,0),([0]*32,1,32)):
            with self.assertRaises(ValueError):vm.Input.make(a,n,0,k)

    def test_elf_rejection(self):
        for raw in (b'',b'not ELF',b'\x7fELF\x02\x01\x01'+bytes(100)):
            with self.assertRaises(ValueError):vm.load_elf(raw)

    def test_patch_boundaries(self):
        code=words(RETURN)
        for pos,value in ((1,0),(4,0),(0,-1),(0,2**32)):
            with self.assertRaises(ValueError):search.patched(code,pos,value)
        self.assertEqual(code,words(RETURN))

    def test_mutation_space_no_identity(self):
        for instruction in (LOAD_A,LOAD_B,RETURN,r_word(0),0x00050513,0x00b50463):
            self.assertNotIn(instruction,search.alternatives(instruction))
        self.assertEqual(search.alternatives(RETURN),[])

    def test_m_extension_cross_decoder(self):
        edge=(0,1,2,31,0x7fffffff,0x80000000,0xffffffff)
        for a in edge:
            for b in edge:
                inp=vm.Input.make([a,b]+[0]*30,0,0)
                for f in range(8):
                    code=words(LOAD_A,LOAD_B,r_word(f,1),RETURN)
                    native,_=self.machine.run(code,vm.CODE,inp);independent=vm.python_run(code,vm.CODE,inp)
                    self.assertEqual(native.observable(),independent.observable(),(a,b,f))
                    self.assertEqual(native.status,0)
                    if b==0 and f in (4,5):self.assertEqual(native.value,0xffffffff)
                    if b==0 and f in (6,7):self.assertEqual(native.value,a)
                    if a==0x80000000 and b==0xffffffff and f==4:self.assertEqual(native.value,a)

    def test_x0_and_store_protection(self):
        # LUI x5,0x1; SW x0,0(x5): guest code must not be writable.
        code=words(0x000012b7,0x0002a023,RETURN);inp=vm.Input.make([0]*32,0,0)
        out,_=self.machine.run(code,vm.CODE,inp)
        self.assertEqual(out.status,2);self.assertEqual(out.observable(),vm.python_run(code,vm.CODE,inp).observable())

    def test_misaligned_load(self):
        code=words(LOAD_A|(1<<20),RETURN);inp=vm.Input.make([0]*32,0,0)
        out,_=self.machine.run(code,vm.CODE,inp);self.assertEqual(out.status,2)

    def test_instruction_budget(self):
        code=words(0x0000006f);inp=vm.Input.make([0]*32,0,0)
        out,trace=self.machine.run(code,vm.CODE,inp,trace=True)
        self.assertEqual((out.status,out.steps),(4,vm.FUEL));self.assertEqual(trace,{0})
        self.assertEqual(out.observable(),vm.python_run(code,vm.CODE,inp).observable())

    def test_illegal_instruction(self):
        inp=vm.Input.make([0]*32,0,0)
        out,_=self.machine.run(words(0xffffffff),vm.CODE,inp);self.assertEqual(out.status,3)

    def test_request_has_no_target_fields(self):
        with self.assertRaises(TypeError):search.Request(code=b'',entry=0,cases=(),expected=(),target=b'')
        with self.assertRaises(ValueError):search.Request(b'',0,(),())

    def test_repair_and_guided_equality(self):
        original=words(LOAD_A,LOAD_B,r_word(0),RETURN)
        faulty=search.patched(original,8,r_word(0,32));copy=faulty[:]
        cases=inputs.random_inputs(409,64);expected=self.machine.batch(original,vm.CODE,cases)
        request=search.Request(faulty,vm.CODE,tuple(cases),tuple(expected))
        all_=search.repair(request,self.machine,guided=False)
        guided=search.repair(request,self.machine,guided=True)
        self.assertEqual(all_.code,guided.code);self.assertEqual(faulty,copy)
        self.assertEqual(all_.status,'candidate');self.assertEqual(all_.assurance,'sample_checked_not_proven')
        self.assertLessEqual(guided.tried,all_.tried)
        new=inputs.random_inputs(8809,128);gold=self.machine.batch(original,vm.CODE,new)
        self.assertEqual(self.machine.check(guided.code,vm.CODE,vm.pack(new,gold)),-1)

    def test_instruction_bytes_read_as_data_survive_pruning(self):
        # LUI x5,0x1; LW a0,20(x5); RET; two NOPs; unexecuted ADD.
        code=words(0x000012b7,0x0142a503,RETURN,0x13,0x13,0x00000533)
        target=search.patched(code,20,0x40000533)
        cases=inputs.random_inputs(917,4);gold=self.machine.batch(target,vm.CODE,cases)
        request=search.Request(code,vm.CODE,tuple(cases),tuple(gold))
        _,seen=self.machine.run(code,vm.CODE,cases[0],trace=True)
        self.assertIn(5,seen)
        whole=search.repair(request,self.machine,guided=False)
        guided=search.repair(request,self.machine,guided=True)
        self.assertEqual(whole.code,guided.code)
        self.assertEqual(guided.status,'candidate')

    def test_clean_program_is_not_edited(self):
        code=words(LOAD_A,LOAD_B,r_word(0),RETURN);cases=inputs.random_inputs(1,4)
        gold=self.machine.batch(code,vm.CODE,cases)
        decision=search.repair(search.Request(code,vm.CODE,tuple(cases),tuple(gold)),self.machine,guided=True)
        self.assertEqual(decision.status,'unchanged');self.assertIsNone(decision.patch)

    def test_inconsistent_spec_abstains(self):
        code=words(LOAD_A,LOAD_B,r_word(0),RETURN);case=inputs.random_inputs(6,1)[0]
        gold=self.machine.batch(code,vm.CODE,[case,case]);gold[1].value^=1
        result=search.repair(search.Request(code,vm.CODE,(case,case),tuple(gold)),self.machine,guided=True)
        self.assertEqual(result.status,'abstain');self.assertEqual(result.code,code)

    @unittest.skipUnless(shutil.which('clang') and shutil.which('ld.lld'),'Clang and LLD required')
    def test_upstream_zlib_against_system_library(self):
        import zlib
        source_path=Path(__file__).resolve().parents[1]/'experiments/compiled_external_v3/adler32.c'
        for opt in ('O0','O2','Os'):
            code,entry,native=vm.compile_kernel(source_path.read_text(),self.root/('upstream-'+opt),opt)
            cases=inputs.random_inputs(60219,64)
            for inp,actual,gold in zip(cases,self.machine.batch(code,entry,cases),vm.reference(native,cases)):
                self.assertEqual(actual.observable(),gold.observable())
                self.assertEqual(actual.value,zlib.adler32(struct.pack('<32I',*inp.a)[:4*inp.n],inp.key))

    @unittest.skipUnless(shutil.which('clang') and shutil.which('ld.lld'),'Clang and LLD required')
    def test_compiled_reference_alias_calls_and_optimization(self):
        for family in ('overlap_move','rare_dispatch','crc_bitwise','insertion_sort'):
            for opt in ('O0','O2','Os'):
                code,entry,native=vm.compile_kernel(corpus.source(family,1),self.root/(family+opt),opt)
                cases=inputs.probe_inputs(code,712,64);gold=vm.reference(native,cases)
                actual=self.machine.batch(code,entry,cases)
                for inp,a,b in zip(cases,actual,gold):
                    self.assertEqual(a.observable(),b.observable(),(family,opt,inp.record()))
                for inp in cases[:4]:
                    a,_=self.machine.run(code,entry,inp)
                    self.assertEqual(a.observable(),vm.python_run(code,entry,inp).observable())
                if family=='rare_dispatch':
                    self.assertTrue(any(i.key==0xc0de0005 for i in cases))
                    hidden=inputs.hidden_inputs(code,351,{inputs.identity(i) for i in cases})
                    self.assertFalse({inputs.identity(i) for i in cases}&{inputs.identity(i) for i in hidden})

if __name__=='__main__':unittest.main()
