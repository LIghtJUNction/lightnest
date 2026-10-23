import unittest,torch,json
from copy import deepcopy
from .model import *
from .prepare import read
from .train import FrozenWorld,legal

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    configure_torch();torch.manual_seed(7);cls.m=HybridNetwork(104729).eval();cls.rows=project(read(ROOT/'data/fit.jsonl')[:2])
 def test_initial_identity(self):
    with torch.no_grad():
     a=self.m.full(self.rows)[-1];b=self.m.red(*common.encode(self.rows,self.m.red.config),rounds=64)
     self.assertTrue(torch.equal(a['scores'],b[0]));self.assertTrue(torch.equal(a['bits'],b[1]))
 def test_link_gradient(self):
    m=deepcopy(self.m).train();rs=read(ROOT/'data/fit.jsonl')[:2];out=m.full(project(rs))
    loss=sum(common.supervised_loss(o['scores'],o['bits'],rs,m.none_index)[0] for o in out);loss.backward()
    self.assertGreater(float(m.link.up.weight.grad.abs().sum()),0)
    self.assertTrue(all(p.grad is None for p in m.red.parameters()))
    self.assertTrue(all(p.grad is None for p in m.green.parameters()))
    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.head.parameters()))
 def test_codec(self):
    for p in (0,17,1023):
     for v in (0,1,64,255):
      a={'position':p,'value':v,'stop':False};self.assertEqual(decode_action(encode_action(a)),a)
    self.assertEqual(decode_action('STOP')['position'],1024)
 def test_reject_invalid_text(self):
    for t in ('PATCH -1 00','PATCH 1024 01','PATCH 1 0f extra','STOP\nexec()','PATCH 01 ff','THINK'):
     with self.assertRaises(ValueError):decode_action(t)
 def test_no_labels_in_forward(self):
    with self.assertRaises(ValueError):self.m.predict([self.rows[0]|{'target':'answer'}])
 def test_default_fixed64(self):
    out=self.m.predict(self.rows);self.assertTrue(all(a['rounds']==64 for a in out))
 def test_real_early_exit(self):
    p=ModePolicy().eval()
    with torch.no_grad():
     for v in p.parameters():v.zero_()
     p.net[-1].bias[1]=10
    out=self.m.predict(self.rows,p);self.assertTrue(all(a['rounds']==4 and len(a['mode_trace'])==1 for a in out))
 def test_live_full_matches_forced(self):
    with torch.no_grad():
     out=self.m.full(self.rows)[-1];expected=[action(out['scores'][j],out['bits'][j],1024) for j in range(2)]
     live=self.m.predict(self.rows,fixed64=True)
     self.assertEqual([(a['position'],a['value']) for a in expected],[(a['position'],a['value']) for a in live])
 def test_proposal_cache_key(self):
    w=FrozenWorld(self.m);w.get(self.rows[0]);w.get(self.rows[0]);self.assertEqual(w.computed_states,1)
    w.get(self.rows[0]|{'command':self.rows[0]['command']+' '});self.assertEqual(w.computed_states,2)
 def test_noop_label(self):
    self.assertFalse(legal({'position':3,'value':0,'stop':False},{'edits':[]}))
 def test_state_feedback_matters(self):
    m=deepcopy(self.m)
    with torch.no_grad():m.link.up.bias.fill_(.04);a=m.full(self.rows);b=m.full(self.rows,linked=False)
    self.assertFalse(torch.equal(a[-1]['h'],b[-1]['h']))
 def test_nonfinite_reject(self):
    m=deepcopy(self.m)
    with torch.no_grad():m.link.up.bias.fill_(float('nan'))
    with self.assertRaises(FloatingPointError):m.predict(self.rows)

if __name__=='__main__':unittest.main()
