import unittest
from lightnest_rgb.structure import StructureNet,facts
class StructureTests(unittest.TestCase):
 def test_facts_and_shape(self):
  raw=b'import x\ndef f(a):\n return x(a)\n';self.assertGreater(facts(raw)[2],0);self.assertEqual(StructureNet()(raw).shape,(1,6))
if __name__=='__main__':unittest.main()
