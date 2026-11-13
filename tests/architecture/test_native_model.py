"""The same shared GRU source executes in Rust and in an explicit Torch model."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import torch
import weightlang as wgt
from lightnest.architecture.native import load_recurrent
from lightnest.architecture.generate import ELFGenerator


class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daemon = os.environ.get('WGTIME_TEST_DAEMON')
        if not daemon:
            raise unittest.SkipTest('set WGTIME_TEST_DAEMON to the matching prebuilt daemon')
        cls.temp=tempfile.TemporaryDirectory(prefix='lightnest-wgtime-')
        directory=Path(cls.temp.name);directory.chmod(0o700)
        cls.process=subprocess.Popen([daemon,'--test-directory',str(directory)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        cls.endpoint=directory/'service.sock'
        for _ in range(100):
            if cls.endpoint.exists():return
            if cls.process.poll() is not None:raise AssertionError(cls.process.stderr.read().decode())
            time.sleep(.02)
        raise AssertionError('wgtime did not start')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:cls.process.kill();cls.process.wait()
        cls.process.stderr.close();cls.temp.cleanup()

    def test_native_and_torch_forward_recurrent_gradients_and_training(self):
        torch.manual_seed(18)
        reference=torch.nn.GRUCell(3,2)
        mapping={}
        for prefix,stem in [('x','ih'),('h','hh')]:
            for part,gate in enumerate('rzn'):
                mapping[prefix+gate+'_weight']=getattr(reference,'weight_'+stem).detach().chunk(3)[part]
                mapping[prefix+gate+'_bias']=getattr(reference,'bias_'+stem).detach().chunk(3)[part]
        portable=load_recurrent(3,2,backend='torch',device='cpu')
        self.assertIsInstance(portable,wgt.Model)
        with torch.no_grad():
            for name,value in mapping.items():
                portable.to_torch().get_parameter(portable.parameter_names[name]).copy_(value)
        with load_recurrent(3,2,endpoint=self.endpoint,optimizer='sgd') as native:
            self.assertIsInstance(native,wgt.Model);self.assertEqual(native.backend,'native')
            native.set_parameters([wgt.Parameter(name,tuple(t.shape),tuple(t.flatten().tolist())) for name,t in mapping.items()])
            native.set_optimizer('sgd')  # explicit parameter replacement resets training state
            initial=torch.tensor([.1,-.2],requires_grad=True)
            a=initial;b=initial.detach().clone().requires_grad_();c=initial.detach().tolist()
            values=[torch.tensor([.2,-.3,.1],requires_grad=True),torch.tensor([.3,.4,-.2],requires_grad=True)]
            for x in values:
                a=reference(x,a);b=portable(x,b)[0];c=native(x.detach().tolist(),c)[0]
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(a.detach(),torch.tensor(c),atol=1e-6,rtol=1e-5)
            a.square().mean().backward();b.square().mean().backward()
            for prefix,stem in [('x','ih'),('h','hh')]:
                for part,gate in enumerate('rzn'):
                    for suffix in ['weight','bias']:
                        expected=getattr(reference,suffix+'_'+stem).grad.chunk(3)[part]
                        actual=portable.to_torch().get_parameter(portable.parameter_names[prefix+gate+'_'+suffix]).grad
                        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            rows=[x.detach().tolist()+([.1,-.2] if i==0 else [0.,0.]) for i,x in enumerate(values)]
            loss=native.train_recurrent(rows,[None,[0.,0.]],input_offset=3,output_offset=0,state_width=2,supervised_offset=0,supervised_width=2,rate=.01)
            self.assertAlmostEqual(loss,float(a.detach().square().mean()),places=6)
            torch.optim.SGD(reference.parameters(),lr=.01).step()
            updated={p.name:torch.tensor(p.data).reshape(p.shape) for p in native.parameters()}
            for prefix,stem in [('x','ih'),('h','hh')]:
                for part,gate in enumerate('rzn'):
                    for suffix in ['weight','bias']:
                        torch.testing.assert_close(updated[prefix+gate+'_'+suffix],getattr(reference,suffix+'_'+stem).detach().chunk(3)[part],atol=1e-6,rtol=1e-5)

    def test_complex_application_explicitly_uses_torch_wgt_object(self):
        model=ELFGenerator(width=8,capacity=16)
        self.assertIsInstance(model,wgt.Model)
        self.assertEqual(model.backend,'torch')
        self.assertIs(model.to_torch(),model)
