import unittest
from lightnest_rgb.any2any import ArtifactCore, ArtifactRequest, KINDS, CopyPasteAction, apply_copy_paste
import hashlib
from lightnest_rgb.artifact_types import detect_kind, inspect_artifact

class Any2AnyTests(unittest.TestCase):
    def test_detects_multiple_real_formats(self):
        self.assertEqual(detect_kind(b'{"x":1}', 'x.json'), 'json')
        self.assertEqual(detect_kind(b'PK\x03\x04rest', 'x.zip'), 'archive')
        self.assertEqual(detect_kind(b'%PDF-1.7', 'x.pdf'), 'pdf')
        self.assertEqual(detect_kind(b'hello,world\n1,2\n', 'x.csv'), 'csv')
    def test_core_has_red_abstract_and_blue_guidance(self):
        model=ArtifactCore(width=32,heads=4,max_bytes=256); request=ArtifactRequest('json','edit')
        encoded,_=model.encode(b'{"x":1}',request); abstract,guidance=model.think(encoded,2); heads=model.heads(encoded,guidance[:,None])
        self.assertEqual(abstract.shape[-1],128); self.assertEqual(heads['blue_logits'].shape[-1],257)
    def test_invalid_request_kind(self):
        with self.assertRaises(ValueError): ArtifactRequest('', 'make')
    def test_native_copy_paste_is_hash_and_range_checked(self):
        raw=b'abcabc'; out=apply_copy_paste(raw,CopyPasteAction(0,3,3),hashlib.sha256(raw).hexdigest())
        self.assertEqual(out,raw)
        with self.assertRaises(ValueError): apply_copy_paste(raw,CopyPasteAction(0,4,3),hashlib.sha256(raw).hexdigest())
        self.assertEqual(apply_copy_paste(raw,CopyPasteAction(0,3,6,'copy_insert'),hashlib.sha256(raw).hexdigest()),b'abcabcabc')
if __name__=='__main__': unittest.main()
