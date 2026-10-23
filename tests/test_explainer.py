import unittest
from lightnest_rgb.explainer import CodeExplainer, summary_text
class ExplainerTests(unittest.TestCase):
 def test_summary_and_forward(self):
  raw=b'def f(x):\n return x+1\n'; self.assertIn('functions f',summary_text(raw,'x.py'))
  m=CodeExplainer(width=32); import torch
  target=torch.tensor([[127,65,128]]); out=m(raw,target[:,:-1]); self.assertEqual(out.shape[-1],129)
if __name__=='__main__':unittest.main()
