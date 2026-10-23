import unittest,torch
from pathlib import Path
from rgb.runtime import configure_torch,common
from rgb.network import load_rgb
from rgb.scripted_core import load_compiled_rgb
ROOT=Path(__file__).resolve().parents[1]
class TestScripted(unittest.TestCase):
 def setUp(self):
  configure_torch();self.m=load_compiled_rgb(ROOT/'models/rgb-104729.pt')
  code=common.assemble('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x10, x1, x2\nebreak\n')
  self.enc=common.encode([{'command':'change every add that contributes to the output to sub','binary':code.hex()}],self.m.red.config)
 def test_no_extra_model_parameters(self):
  self.assertEqual(sum(p.numel() for p in self.m.parameters()),157214)
 def test_auxiliary_nan_isolation(self):
  ref=self.m(*self.enc,mode='red_only')
  with torch.no_grad():self.m.green.queries.fill_(float('nan'))
  out=self.m(*self.enc,mode='red_only');self.assertTrue(torch.equal(ref['scores'],out['scores']))
 def test_red_nan_rejected(self):
  with torch.no_grad():self.m.red.update.weight_ih[0,0]=float('nan')
  with self.assertRaises(FloatingPointError):self.m(*self.enc,mode='red_only')
 def test_parameter_replacement_rejected(self):
  self.m.red.definition_message.weight=torch.nn.Parameter(self.m.red.definition_message.weight.clone(),requires_grad=False)
  with self.assertRaises(RuntimeError):self.m(*self.enc,mode='red_only')
 def test_training_rejected(self):
  self.m.train()
  with self.assertRaises(RuntimeError):self.m(*self.enc,mode='red_only')
 def test_same_logits(self):
  old=load_rgb(ROOT/'models/rgb-104729.pt')
  a=old(*self.enc,mode='red_only');b=self.m(*self.enc,mode='red_only')
  self.assertTrue(torch.equal(a['scores'],b['scores']));self.assertTrue(torch.equal(a['bits'],b['bits']))
