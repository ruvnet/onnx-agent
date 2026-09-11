import unittest,subprocess,sys,json
from onnx_agent.pipeline import *
class PipelineTests(unittest.TestCase):
 def test_real_cpu_parity(self):
  r=qualify(20);self.assertTrue(r['parityPass'],r);self.assertLess(r['results']['int8']['bytes'],r['results']['float32']['bytes'])
 def test_reference(self):
  d,w,b=fixture();x=np.ones((8,64),dtype=np.float32);np.testing.assert_allclose(session(d,'CPUExecutionProvider').run(None,{'input':x})[0],x@w+b,atol=1e-6)
 def test_bounds(self):
  for i in [0,1001,True,-1]:
   with self.assertRaises(ValueError): qualify(i)
 def test_external_rejected(self):
  d,_,_=fixture();m=onnx.load_model_from_string(d);m.graph.initializer[0].data_location=TensorProto.EXTERNAL
  with self.assertRaises(ValueError): validate(m.SerializeToString())
 def test_operator_rejected(self):
  d,_,_=fixture();m=onnx.load_model_from_string(d);m.graph.node[0].op_type='Loop'
  with self.assertRaises(ValueError):validate(m.SerializeToString())
 def test_size(self):
  with self.assertRaises(ValueError):validate(b'x'*(MAX_BYTES+1))
 def test_gpu_rejected(self):
  with self.assertRaises(ValueError):session(fixture()[0],'CUDAExecutionProvider')
 def test_cli(self):
  r=subprocess.run([sys.executable,'-m','onnx_agent','qualify','--iterations','2'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertTrue(json.loads(r.stdout)['parityPass'])

 def test_shape_product_overflow_rejected_before_checker(self):
  from unittest.mock import patch
  d,_,_=fixture();m=onnx.load_model_from_string(d);del m.graph.initializer[0].dims[:];m.graph.initializer[0].dims.extend([1024]*7)
  with patch('onnx.checker.check_model') as checker:
   with self.assertRaisesRegex(ValueError,'tensor dimensions'):validate(m.SerializeToString())
   checker.assert_not_called()

 def test_optimization_parity(self):
  d,w,b=fixture();x=np.ones((8,64),dtype=np.float32)
  a=session(d,'CPUExecutionProvider',False).run(None,{'input':x})[0]
  z=session(d,'CPUExecutionProvider',True).run(None,{'input':x})[0]
  np.testing.assert_allclose(a,z,atol=1e-6)
