import asyncio, unittest
from pathlib import Path
from lightnest.self_hosting.self_agent import SelfAgent
class SelfAgentTests(unittest.TestCase):
 def test_generic_tool_inspection(self):
  report=asyncio.run(SelfAgent().inspect(Path('lightnest')));self.assertIn('module_graph',report['tools']);self.assertEqual(report['errors'],[]);self.assertIn('architecture/model.py',report['graph'])
if __name__=='__main__':unittest.main()
