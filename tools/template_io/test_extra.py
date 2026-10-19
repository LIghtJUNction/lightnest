import json,os,random,shutil,tempfile,unittest
from pathlib import Path
from template_io import *
from topology_cache import topology_bytes
from file_patch import create,restore

class ExtraTests(unittest.TestCase):
 def test_chunk_roundtrip(self):
  r=random.Random(917)
  for kind in ['fixed','gear']:
   for _ in range(25):
    b=r.randbytes(8192);t=b[:100]+r.randbytes(3)+b[100:]
    p,s=chunk_packet(b,t,kind,256);self.assertEqual(restore_chunks(b,p),t)
 def test_chunk_wrong_base(self):
  p,_=chunk_packet(b'abc',b'xyz')
  with self.assertRaises(ValueError):restore_chunks(b'wrong',p)
 def test_chunk_bomb(self):
  p,_=chunk_packet(b'a'*1024,b'a'*1024)
  with self.assertRaises(ValueError):restore_chunks(b'a'*1024,p,max_output=5)
 def test_topology_preserves_return(self):
  a=(0x00050513).to_bytes(4,'little');b=(0x00058513).to_bytes(4,'little')
  self.assertNotEqual(topology_bytes(a),topology_bytes(b))
 def test_topology_normalizes_alu_not_memory(self):
  self.assertEqual(topology_bytes((0x401080b3).to_bytes(4,'little')),topology_bytes((0x001080b3).to_bytes(4,'little')))
  self.assertNotEqual(topology_bytes((0x04002083).to_bytes(4,'little')),topology_bytes((0x04402083).to_bytes(4,'little')))
 @unittest.skipUnless(shutil.which('zstd'),'zstd CLI unavailable')
 def test_file_patch_with_hash_checks(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);b=d/'base';t=d/'target';p=d/'package';o=d/'result';r=random.Random(825)
   base=r.randbytes(65536);target=base[:8192]+b'new bytes'+base[8192:]
   b.write_bytes(base);t.write_bytes(target);m=create(t,p,b)
   self.assertEqual(m['kind'],'delta');self.assertFalse(m['executed'])
   restore(p,o,b);self.assertEqual(o.read_bytes(),target)
   with self.assertRaises(FileExistsError):restore(p,o,b)
   b.write_bytes(b'wrong')
   with self.assertRaises(ValueError):restore(p,d/'fail',b)
   self.assertFalse((d/'fail').exists())
 @unittest.skipUnless(shutil.which('zstd'),'zstd CLI unavailable')
 def test_file_full_and_size_guard(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);t=d/'target';p=d/'package';t.write_bytes(b'payload'*200)
   self.assertEqual(create(t,p)['kind'],'full')
   with self.assertRaises(ValueError):restore(p,d/'fail',max_bytes=1)
   restore(p,d/'good');self.assertEqual((d/'good').read_bytes(),t.read_bytes())
 @unittest.skipUnless(shutil.which('zstd'),'zstd CLI unavailable')
 def test_random_delta_falls_back_full(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);r=random.Random(991);t=d/'target';b=d/'base';b.write_bytes(r.randbytes(4096));t.write_bytes(r.randbytes(4096))
   self.assertEqual(create(t,d/'package',b)['kind'],'full')
if __name__=='__main__':unittest.main(verbosity=2)
