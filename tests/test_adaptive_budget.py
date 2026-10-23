import unittest
from lightnest_rgb.model import RGBPolicy,RGBConfig
class AdaptiveBudgetTests(unittest.TestCase):
 def test_easy_and_hard_inputs_choose_different_budgets(self):
  m=RGBPolicy(RGBConfig(width=32,heads=4,landmarks=2,detail_k=2,red_depths=(2,4))).eval()
  easy=m.adaptive_depth(b'a'*32); hard=m.adaptive_depth(bytes(range(256))*256)
  self.assertLessEqual(easy,hard);self.assertGreaterEqual(hard,2)
if __name__=='__main__':unittest.main()
