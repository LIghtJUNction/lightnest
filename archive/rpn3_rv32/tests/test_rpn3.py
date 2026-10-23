import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import torch

from lightnest_next.data import output_directory
from lightnest_next.isa import execute_b, graph
from lightnest_next.model import (Batch, Config, PatchNetwork, load_model,
                                 save_model, supervised_loss, validate_request)
from lightnest_next.training import comparison, score_predictions
from tools.check_source_tree import check


def instruction(opcode=0x33, rd=10, a=1, b=2, funct7=0):
    return (funct7<<25) | (b<<20) | (a<<15) | (rd<<7) | opcode


def example(extra=False):
    words = [0x00002083, 0x00402103, instruction()]
    if extra:
        words += [instruction(rd=11, a=1, b=2)] * 3
    words.append(0x00100073)
    raw = b"".join(struct.pack("<I", w) for w in words)
    target = bytearray(raw); target[11] ^= 64
    return {"command":"change add to sub", "binary":raw.hex(), "target":target.hex()}


def request(row):
    return {k:row[k] for k in ("command","binary")}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(71)
        return PatchNetwork(Config(width=32,heads=4,rounds=2))

    def test_configuration_boundaries(self):
        for args in ({"width":0},{"width":33},{"rounds":True},{"max_bytes":1028},{"max_edits":9},{"rounds":65}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Config(**args)

    def test_labels_cannot_enter_inference(self):
        model = self.model().eval()
        with self.assertRaises(ValueError):
            model.predict([example()])

    def test_input_validation(self):
        row = request(example())
        for changes in ({"command":"fix this program"},{"binary":""},{"binary":"zz"},
                        {"binary":"73001000 "},{"binary":"00000000"},{"binary":"7300100073001000"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_request(row | changes, Config())

    def test_absolute_memory_boundary(self):
        row=request(example())
        raw=struct.pack("<II",0x0000a503,0x00100073)
        with self.assertRaises(ValueError):
            validate_request(row | {"binary":raw.hex()},Config())

    def test_train_eval_boundary(self):
        with self.assertRaises(RuntimeError):
            self.model().predict([request(example())])

    def test_padding_does_not_change_predictions(self):
        model = self.model().eval()
        a,b = request(example()),request(example(True))
        single=model(Batch.encode([a],model.config))
        together=model(Batch.encode([a,b],model.config))
        torch.testing.assert_close(single[0][0],together[0][0,:16],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(single[1][0],together[1][0,:16],rtol=1e-4,atol=1e-5)
        self.assertTrue(torch.isneginf(together[0][0,16:]).all())

    def test_forward_backward_has_finite_gradients(self):
        model = self.model();rows=[example(),example(True)]
        batch=Batch.encode([request(r) for r in rows],model.config)
        z,b=model(batch); loss=supervised_loss(z,b,batch,rows);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(model.update.weight_hh.grad.abs().sum()),0)

    def test_every_required_edit_receives_feedback(self):
        row=example(True);target=bytearray.fromhex(row['target']);target[15]^=64;row['target']=target.hex()
        batch=Batch.encode([request(row)],Config())
        logits=torch.full((1,28),-6.,requires_grad=True)
        bits=torch.zeros(1,28,8,requires_grad=True)
        loss=supervised_loss(logits,bits,batch,[row]);loss.backward()
        self.assertLess(float(logits.grad[0,11]),0)
        self.assertLess(float(logits.grad[0,15]),0)
        self.assertNotEqual(float(bits.grad[0,11,6]),0)
        self.assertNotEqual(float(bits.grad[0,15,6]),0)

    def test_target_length_and_budget_rejected(self):
        row=example();batch=Batch.encode([request(row)],Config())
        z=torch.zeros(1,16);b=torch.zeros(1,16,8)
        for target in ('00', 'ff'*16):
            with self.assertRaises(ValueError):
                supervised_loss(z,b,batch,[row|{'target':target}])

    def test_checkpoint_round_trip_is_exact(self):
        model=self.model().eval();row=request(example())
        before=model.predict([row])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';save_model(path,model,{'status':'test fixture'})
            self.assertEqual(before,load_model(path).predict([row]))
            with self.assertRaises(FileExistsError):
                save_model(path,model,{})

    def test_output_budget_and_no_progress_fail_closed(self):
        model=self.model().eval();row=request(example())
        with torch.no_grad():
            model.edit.weight.zero_();model.edit.bias.fill_(20)
        pred=model.predict([row])[0]
        self.assertEqual(pred['status'],'budget')
        self.assertFalse(pred['stop']);self.assertEqual(pred['final'],row['binary'])

    def test_all_negative_is_explicit_stop_without_edits(self):
        model=self.model().eval();row=request(example())
        with torch.no_grad():
            model.edit.weight.zero_();model.edit.bias.fill_(-20)
        pred=model.predict([row])[0]
        self.assertTrue(pred['stop']);self.assertEqual(pred['patches'],[])
        self.assertEqual(pred['final'],row['binary'])

    def test_nonfinite_output_is_rejected(self):
        model=self.model().eval()
        with torch.no_grad():
            model.edit.bias.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            model.predict([request(example())])


class EvidenceTests(unittest.TestCase):
    def test_interpreter_and_dependency_edges(self):
        row=example();raw=bytes.fromhex(row['binary'])
        self.assertEqual(execute_b(raw,[8,3]),11)
        self.assertEqual(execute_b(bytes.fromhex(row['target']),[8,3]),5)
        self.assertIn((2,0),graph(raw));self.assertIn((2,1),graph(raw));self.assertIn((3,2),graph(raw))

    def test_success_needs_both_stop_and_expected_execution(self):
        row=example()|{'id':'x','family':'simple','edits':[{'offset':11,'value':64}],'cases':[[[8,3],5]]}
        pred={'final':row['target'],'stop':False,'status':'budget','patches':[{'offset':11,'value':64}]}
        self.assertFalse(score_predictions([row],[pred],execute_b)[0]['strict'])
        self.assertTrue(score_predictions([row],[pred|{'stop':True}],execute_b)[0]['strict'])
        row['cases']=[[[8,3],99]]
        self.assertFalse(score_predictions([row],[pred|{'stop':True}],execute_b)[0]['strict'])

    def test_comparison_preserves_ids_and_input_identity(self):
        base=[{'id':'x','initial_sha256':'a','strict':True,'false_edit':False}]
        candidate=[base[0]|{'strict':False,'false_edit':True}]
        cmp=comparison(base,candidate)
        self.assertEqual(cmp['lost_success_ids'],['x']);self.assertEqual(cmp['added_false_edit_ids'],['x'])
        for change in ({'id':'y'},{'initial_sha256':'b'}):
            with self.assertRaises(ValueError):
                comparison(base,[candidate[0]|change])

    def test_outputs_cannot_be_written_into_repo(self):
        with self.assertRaises(ValueError):
            output_directory(Path(__file__).resolve().parents[1]/'results')

    def test_untracked_artifacts_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);subprocess.run(['git','init','-q',temp],check=True)
            (root/'bad.bin').write_bytes(b'not source\x00')
            self.assertTrue(check(root)[1])

    def test_deleted_tracked_sources_and_new_uv_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);subprocess.run(['git','init','-q',temp],check=True)
            path=root/'old.py';path.write_text('x=1\n')
            subprocess.run(['git','-C',temp,'add','old.py'],check=True);path.unlink()
            (root/'.python-version').write_text('3.13\n');(root/'uv.lock').write_text('version = 1\n')
            self.assertEqual(check(root),(2,[]))


if __name__ == '__main__':
    unittest.main()
