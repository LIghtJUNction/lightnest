import unittest
from pathlib import Path
from lightnest.self_hosting.unified import RGBArtifactAgent

class UnifiedSelfTests(unittest.TestCase):
    def test_unified_agent_checks_current_package(self):
        report=RGBArtifactAgent(workdir='/tmp/lightnest-unified-test').inspect_project(Path('lightnest'))
        self.assertIn('module_graph',report['tools']); self.assertIn('model.py',report['graph'])

if __name__=='__main__': unittest.main()
