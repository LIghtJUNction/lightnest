import hashlib,io,random,unittest
from dataclasses import replace
from template_io import *

class CodecTests(unittest.TestCase):
 def test_uint(self):
  for n in [0,1,127,128,1024,2**32,2**63-1]:self.assertEqual(read_uint(io.BytesIO(uint(n))),n)
 def test_uint_invalid(self):
  for raw in [b'',b'\x80',b'\xff'*10]:
   with self.assertRaises(ValueError):read_uint(io.BytesIO(raw))
  with self.assertRaises(ValueError):uint(-1)
 def test_random_delta(self):
  r=random.Random(21453)
  for _ in range(1000):
   a=r.randbytes(r.randrange(0,2048));b=bytearray(a)
   for _ in range(r.randrange(0,12)):
    if b:b[r.randrange(len(b))]=r.randrange(256)
   self.assertEqual(apply_delta(a,make_delta(a,bytes(b))),bytes(b))
 def test_size_change(self):
  for a,b in [(b'abc',b'abcccc'),(b'abc',b''),(b'',b'abc')]:self.assertEqual(apply_delta(a,make_delta(a,b)),b)
 def test_patch_bounds(self):
  with self.assertRaises(ValueError):apply_delta(b'ab',b'\x01\x01\x04\x01x')
 def test_packet_base_binding(self):
  p=transport_packet(b'abcd',b'abxd')
  self.assertEqual(restore_packet(b'abcd',p),b'abxd')
  with self.assertRaises(ValueError):restore_packet(b'zzzz',p)
 def test_packet_corruption(self):
  p=transport_packet(b'abcd',b'abxd')
  with self.assertRaises(ValueError):restore_packet(b'abcd',p[:-1]+bytes([p[-1]^1]))
 def test_template_roundtrip(self):
  a=(0x001080b3).to_bytes(4,'little');b=(0x401080b3).to_bytes(4,'little')
  self.assertEqual(compression_template_key(a),compression_template_key(b))
  rows=[CoreRow(hashlib.sha256(a).digest(),'edit',a,b),CoreRow(hashlib.sha256(b).digest(),'stop',b,b)]
  for templates in [False,True]:self.assertEqual(list(unpack_core(pack_core(rows,templates)[0])),rows)
 def test_archive_corruption(self):
  data=pack_core([],True)[0]
  with self.assertRaises(ValueError):list(unpack_core(data[:-1]+b'!'))
 def test_archive_bad_template(self):
  body=b'LNTP1C\x01xR'+b'0'*32+b'\x00\x01\x01\x00\x01\x00'
  with self.assertRaises(ValueError):list(unpack_core(body+hashlib.sha256(body).digest()))
 def test_cache_hit(self):
  c=ExactRequestCache();ctx=CacheContext('t','m','r');calls=[]
  for _ in range(4):self.assertEqual(c.get_or_compute(ctx,'x',b'abc',lambda:calls.append(1) or b'answer'),b'answer')
  self.assertEqual(len(calls),1);self.assertEqual(c.hits,3)
 def test_cache_context_invalidation(self):
  c=ExactRequestCache();ctx=CacheContext('t','m','r')
  c.get_or_compute(ctx,'x',b'abc',lambda:b'old')
  for other in [replace(ctx,tenant='u'),replace(ctx,model_sha256='m2'),replace(ctx,runtime_sha256='r2'),replace(ctx,rounds=32),replace(ctx,max_edits=2)]:
   self.assertEqual(c.get_or_compute(other,'x',b'abc',lambda:b'new'),b'new')
  self.assertEqual(c.hits,0)
 def test_cache_different_return_or_command(self):
  c=ExactRequestCache();ctx=CacheContext('t','m','r')
  self.assertEqual(c.get_or_compute(ctx,'add',b'left',lambda:b'A'),b'A')
  self.assertEqual(c.get_or_compute(ctx,'add',b'right',lambda:b'B'),b'B')
  self.assertEqual(c.get_or_compute(ctx,'sub',b'left',lambda:b'C'),b'C')
  self.assertEqual(c.hits,0)
 def test_cache_bounds(self):
  c=ExactRequestCache(max_entries=1);ctx=CacheContext('t','m','r')
  for x in [b'a',b'b',b'a']:c.get_or_compute(ctx,'x',x,lambda:b'v')
  self.assertEqual(c.hits,0);self.assertEqual(len(c.data),1)
 def test_zero_progress_and_literal(self):
  for delta in [b'\x00x',b'\x04',b'\x02\x05abc']:
   with self.assertRaises(ValueError):apply_delta(b'foo',delta)

if __name__=='__main__':unittest.main(verbosity=2)
