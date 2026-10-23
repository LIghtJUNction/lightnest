import unittest
import torch
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
        self.assertEqual(heads['budget_logits'].shape[-1],len(model.red_depths)+1)

    def test_generic_summary_uses_learned_budget(self):
        model=ArtifactCore(width=32,heads=4,max_bytes=256).eval(); request=ArtifactRequest('text','understand')
        with torch.no_grad():
            model.budget[-1].weight.zero_(); model.budget[-1].bias.fill_(-10); model.budget[-1].bias[2]=10
        _, result=model.summarize(b'hello world',request)
        self.assertEqual(result['budget']['depth'], model.red_depths[1])
    def test_invalid_request_kind(self):
        with self.assertRaises(ValueError): ArtifactRequest('', 'make')
    def test_native_copy_paste_is_hash_and_range_checked(self):
        raw=b'abcabc'; out=apply_copy_paste(raw,CopyPasteAction(0,3,3),hashlib.sha256(raw).hexdigest())
        self.assertEqual(out,raw)
        with self.assertRaises(ValueError): apply_copy_paste(raw,CopyPasteAction(0,4,3),hashlib.sha256(raw).hexdigest())
        self.assertEqual(apply_copy_paste(raw,CopyPasteAction(0,3,6,'copy_insert'),hashlib.sha256(raw).hexdigest()),b'abcabcabc')

    def test_model_copy_paste_proposal_is_publishable(self):
        model=ArtifactCore(width=32,heads=4,max_bytes=256).eval(); raw=b'copy me copy me'
        request=ArtifactRequest('text','copy_paste')
        action,_=model.propose_copy_paste(raw,request)
        result=apply_copy_paste(raw,action,hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(result),len(raw))
if __name__=='__main__': unittest.main()
