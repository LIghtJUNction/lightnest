import asyncio, unittest
from pathlib import Path
from lightnest_rgb.self_agent import SelfAgent
class SelfAgentTests(unittest.TestCase):
 def test_generic_tool_inspection(self):
  report=asyncio.run(SelfAgent().inspect(Path('lightnest_rgb')));self.assertIn('module_graph',report['tools']);self.assertEqual(report['errors'],[]);self.assertIn('model.py',report['graph'])
if __name__=='__main__':unittest.main()
