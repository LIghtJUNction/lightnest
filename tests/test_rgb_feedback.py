import unittest
from pathlib import Path
import torch
from lightnest.feedback import outcome_reward, reinforce_loss, transition_reward
from lightnest.execute import Case, ExecutableRunner
from lightnest.iterate import Example

class FeedbackTests(unittest.TestCase):
    def test_invalid_entry_is_typed_without_child(self):
        raw = bytearray(Path('/usr/bin/true').read_bytes()); raw[24:32] = (0x400000000000).to_bytes(8,'little')
        runner=ExecutableRunner(Path('/tmp/lightnest-feedback-test'))
        example=Example(Case(), {'status':'ok','returncode':0,'stdout':'','stderr':''})
        result=outcome_reward(bytes(raw), bytes(raw), [example], runner)
        self.assertIn('invalid_entry_or_elf', result.error_kinds)
        self.assertEqual(result.passed_after, 0)
        self.assertEqual(runner.native_calls, 0)
    def test_reinforce_loss_has_gradient(self):
        logits=torch.zeros(4,requires_grad=True); bits=torch.zeros(4,8,requires_grad=True)
        out={'logits':logits,'bits':bits}; action={'offset':1,'value':3,'stop':False}
        loss=reinforce_loss(out,action,1.0); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertIsNotNone(logits.grad); self.assertIsNotNone(bits.grad)
    def test_transition_reward_prefers_visible_progress(self):
        self.assertGreater(transition_reward(1, 2, True, True), transition_reward(1, 1, True, True))
        self.assertLess(transition_reward(2, 1, True, False), 0)
if __name__=='__main__': unittest.main()
