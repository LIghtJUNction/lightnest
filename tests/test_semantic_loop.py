import asyncio,unittest
from pathlib import Path
from lightnest_rgb.semantic_loop import SemanticLoop
class SemanticLoopTests(unittest.TestCase):
 def test_generic_loop(self):
  r=asyncio.run(SemanticLoop().inspect(Path('lightnest_rgb')));self.assertGreater(r['tool_calls'],0);self.assertFalse(r['errors'])
if __name__=='__main__':unittest.main()
