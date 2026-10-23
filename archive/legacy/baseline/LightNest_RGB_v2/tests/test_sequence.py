import unittest,torch
from rgb.sequence import *
from rgb.runtime import common

def stop():return {'position':1024,'value':None,'stop':True,'confidence':1.,'rounds':64}
def edit(p,v):return {'position':p,'value':v,'stop':False,'confidence':1.,'rounds':64}
class TestSequence(unittest.TestCase):
 def setUp(self):
  self.raw=common.assemble('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x10, x1, x2\nebreak\n')
  self.r={'command':'change every add that contributes to the output to sub','binary':self.raw.hex()};self.pos=11
 def test_no_extra_inputs(self):
  with self.assertRaises(ValueError):inputs_only([self.r|{'target':'00'}])
 def test_invalid_length(self):
  with self.assertRaises(ValueError):inputs_only([self.r|{'binary':'00'}])
 def test_stop_consumes_no_edit(self):
  s=new_state(self.r);commit(s,stop(),origin='red');self.assertEqual(s['code'],self.raw);self.assertTrue(s['stop'])
 def test_terminal_reject(self):
  s=new_state(self.r);commit(s,stop(),origin='red')
  with self.assertRaises(ValueError):commit(s,stop(),origin='red')
 def test_global_budget(self):
  s=new_state(self.r)
  for i in range(9):commit(s,edit(self.pos,s['code'][self.pos]^64),origin='red')
  self.assertEqual(s['status'],'budget');self.assertEqual(s['code'],self.raw)
 def test_no_progress(self):
  s=new_state(self.r);commit(s,edit(self.pos,self.raw[self.pos]),origin='red');self.assertEqual(s['status'],'no_progress')
 def test_outside_pointer(self):
  s=new_state(self.r);commit(s,edit(900,10),origin='red');self.assertEqual(s['status'],'padded_pointer')
 def test_wrong_stop_corrected(self):
  blue=lambda rows:[stop() for r in rows]
  def red(rows):return [edit(self.pos,bytes.fromhex(r['binary'])[self.pos]^64) if bytes.fromhex(r['binary'])[self.pos]==self.raw[self.pos] else stop() for r in rows]
  got,d,meta=verified_drafts(blue,red,[self.r]);ref=continue_red(red,[self.r],[new_state(self.r)])
  self.assertEqual([signature(a) for a in got[0]['actions']],[signature(a) for a in ref[0]['actions']]);self.assertEqual(meta['mismatches'],1)
 def test_reject_bad_suffix(self):
  blue=lambda rows:[edit(0,255) for r in rows]
  red=lambda rows:[stop() for r in rows]
  got,d,meta=verified_drafts(blue,red,[self.r]);self.assertTrue(got[0]['stop']);self.assertEqual(got[0]['code'],self.raw);self.assertNotEqual(d[0]['state']['code'],self.raw)
 def test_matching_two_actions(self):
  def pred(rows):return [edit(self.pos,bytes.fromhex(r['binary'])[self.pos]^64) if bytes.fromhex(r['binary'])[self.pos]==self.raw[self.pos] else stop() for r in rows]
  st,d,meta=verified_drafts(pred,pred,[self.r]);self.assertEqual(meta['verified_prefix_actions'],2);self.assertTrue(st[0]['stop'])
 def test_nonfinite_blue_red_fallback(self):
  def bad(rows):raise FloatingPointError('test')
  st,d,meta=verified_drafts(bad,lambda rows:[stop() for r in rows],[self.r]);self.assertTrue(st[0]['stop']);self.assertEqual(d[0]['state']['status'],'fast_nonfinite')
 def test_stats_ablation(self):
  m=SequenceRisk(True);x=torch.randn(3,144);a=m(x);x[:,:128]+=100;self.assertTrue(torch.equal(a,m(x)))
 def test_risk_negative_threshold_falls_back(self):
  m=SequenceRisk();st,ds,meta=risk_drafts(lambda rows:[stop() for r in rows],lambda rows:[stop() for r in rows],m,[self.r],-1)
  self.assertEqual(meta['accepted_drafts'],0);self.assertEqual(st[0]['actions'][0]['origin'],'red')
if __name__=='__main__':unittest.main()

class TestPrefixFallback(unittest.TestCase):
 def setUp(self):
  self.raw=common.assemble('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nadd x10, x3, x2\nebreak\n')
  self.row={'command':'change every add that contributes to the output to sub','binary':self.raw.hex()}
 def red(self,rows):
  outs=[]
  for r in rows:
   b=bytes.fromhex(r['binary']);p=next((p for p in (11,15) if b[p]==self.raw[p]),None)
   outs.append(stop() if p is None else edit(p,b[p]^64))
  return outs
 def test_one_correct_then_wrong_stop(self):
  def blue(rows):
   return [edit(11,self.raw[11]^64) if bytes.fromhex(r['binary'])[11]==self.raw[11] else stop() for r in rows]
  st,ds,c=verified_drafts(blue,self.red,[self.row]);ref=continue_red(self.red,[self.row],[new_state(self.row)])
  self.assertEqual(st[0]['code'],ref[0]['code']);self.assertEqual([a['step'] for a in st[0]['actions']],[0,1,2]);self.assertEqual(c['verified_prefix_actions'],1)
 def test_rejected_full_draft_restarts_original(self):
  risk=SequenceRisk()
  for p in risk.parameters():p.data.zero_()
  st,ds,c=risk_drafts(lambda rows:[edit(0,255) for r in rows],self.red,risk,[self.row],0.1)
  self.assertEqual(c['accepted_drafts'],0);self.assertTrue(st[0]['stop']);self.assertEqual(st[0]['actions'][0]['position'],11)
 def test_red_row_budget_does_not_reset(self):
  calls=[]
  def endlessly(rows):
   calls.extend(rows);return [edit(11,bytes.fromhex(r['binary'])[11]^64) for r in rows]
  st,ds,c=verified_drafts(endlessly,endlessly,[self.row],draft_steps=4)
  self.assertEqual(len(st[0]['actions']),9);self.assertEqual(st[0]['status'],'budget')
