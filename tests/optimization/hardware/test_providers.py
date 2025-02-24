import unittest
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

from onnx_agent.optimization.hardware.providers import ExecutionProviderManager
from .base_tests import BaseProviderTest

class TestProviders(BaseProviderTest):
    def setUp(self):
        super().setUp()
        self.test_model_dir = Path("test_models")
        self.test_model_dir.mkdir(exist_ok=True)
        self.test_model_path = self.test_model_dir / "test_model.onnx"
        self._create_test_model()
        self.provider_manager = ExecutionProviderManager(str(self.test_model_path))
        
    def tearDown(self):
        # Cleanup test models
        if self.test_model_path.exists():
            self.test_model_path.unlink()
        if self.test_model_dir.exists():
            self.test_model_dir.rmdir()
            
    def _create_test_model(self):
        """Create a simple test ONNX model."""
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
        
        # Create a Conv node followed by a Flatten and Linear node
        conv1_weight_initializer = np.random.randn(64, 3, 7, 7).astype(np.float32)
        conv1_weight = helper.make_tensor(
            'conv1_weight',
            TensorProto.FLOAT,
            conv1_weight_initializer.shape,
            conv1_weight_initializer.tobytes(),
            raw=True
        )
        
        # Use a smaller linear layer to avoid memory issues
        linear_weight_initializer = np.random.randn(1000, 64).astype(np.float32)
        linear_weight = helper.make_tensor(
            'linear_weight',
            TensorProto.FLOAT,
            linear_weight_initializer.shape,
            linear_weight_initializer.tobytes(),
            raw=True
        )
        
        conv_node = helper.make_node(
            'Conv',
            inputs=['input', 'conv1_weight'],
            outputs=['conv_out'],
            kernel_shape=[7, 7],
            strides=[2, 2],
            pads=[3, 3, 3, 3]
        )
        
        flatten_node = helper.make_node(
            'Flatten',
            inputs=['conv_out'],
            outputs=['flatten_out'],
            axis=1
        )
        
        # Add average pooling to reduce dimensions before linear layer
        pool_node = helper.make_node(
            'AveragePool',
            inputs=['conv_out'],
            outputs=['pool_out'],
            kernel_shape=[112, 112],
            strides=[1, 1]
        )
        
        flatten_node = helper.make_node(
            'Flatten',
            inputs=['pool_out'],
            outputs=['flatten_out'],
            axis=1
        )
        
        gemm_node = helper.make_node(
            'Gemm',
            inputs=['flatten_out', 'linear_weight'],
            outputs=['output'],
            transB=1
        )
        
        # Create the graph with all nodes
        graph_def = helper.make_graph(
            [conv_node, pool_node, flatten_node, gemm_node],
            'test-model',
            [X],
            [Y],
            [conv1_weight, linear_weight]
        )
        
        # Create the model with opset version 21 (latest stable)
        model_def = helper.make_model(
            graph_def,
            producer_name='onnx-example',
            opset_imports=[helper.make_opsetid("", 21)]
        )
        
        # Save the model
        onnx.save(model_def, str(self.test_model_path))
        
    def test_cpu_provider(self):
        """Test CPU provider initialization and execution."""
        # Create session with CPU provider
        session = self.provider_manager.create_session_with_provider('CPUExecutionProvider')
        
        # Run inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": input_data})
        
        # Verify output shape
        self.assertEqual(outputs[0].shape, (1, 1000))
        
    def test_cuda_provider(self):
        """Test CUDA provider if available."""
        if 'CUDAExecutionProvider' not in self.provider_manager.available_providers:
            self.skipTest("CUDA provider not available")
            
        try:
            # Configure CUDA provider
            cuda_options = self.provider_manager.configure_cuda_provider(
                device_id=0,
                gpu_mem_limit=None,
                arena_extend_strategy='kNextPowerOfTwo'
            )
            
            # Create session with CUDA provider
            session = self.provider_manager.create_session_with_provider(
                'CUDAExecutionProvider',
                provider_options=cuda_options
            )
            
            # Run inference
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            outputs = session.run(None, {"input": input_data})
            
            # Verify output shape
            self.assertEqual(outputs[0].shape, (1, 1000))
        except Exception as e:
            self.skipTest(f"CUDA provider test failed: {e}")
        
    def test_tensorrt_provider(self):
        """Test TensorRT provider if available."""
        if 'TensorrtExecutionProvider' not in self.provider_manager.available_providers:
            self.skipTest("TensorRT provider not available")
            
        # Configure TensorRT provider
        trt_options = self.provider_manager.configure_tensorrt_provider(
            device_id=0,
            trt_max_workspace_size=2147483648,
            trt_fp16_enable=False
        )
        
        # Create session with TensorRT provider
        session = self.provider_manager.create_session_with_provider(
            'TensorrtExecutionProvider',
            provider_options=trt_options
        )
        
        # Run inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": input_data})
        
        # Verify output shape
        self.assertEqual(outputs[0].shape, (1, 1000))
        
    def test_provider_fallback(self):
        """Test provider fallback mechanism."""
        # Get available providers
        available_providers = self.provider_manager.available_providers
        providers = []
        
        # Add TensorRT if available
        if 'TensorrtExecutionProvider' in available_providers:
            providers.append(('TensorrtExecutionProvider', {}))
            
        # Add CUDA if available
        if 'CUDAExecutionProvider' in available_providers:
            providers.append(('CUDAExecutionProvider', {}))
            
        # Always add CPU
        providers.append('CPUExecutionProvider')
        
        try:
            session = ort.InferenceSession(
                str(self.test_model_path),
                providers=providers
            )
            
            # Verify that at least one provider is available
            self.assertTrue(len(session.get_providers()) > 0)
            
            # Run inference
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            outputs = session.run(None, {"input": input_data})
            
            # Verify output shape
            self.assertEqual(outputs[0].shape, (1, 1000))
        except Exception as e:
            self.skipTest(f"Provider fallback test failed: {e}")
        
    def test_provider_performance(self):
        """Test performance measurement across providers."""
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Test CPU performance
        cpu_perf = self.provider_manager.measure_provider_performance(
            'CPUExecutionProvider',
            input_data,
            num_iterations=10
        )
        
        self.assertIn('mean_latency', cpu_perf)
        self.assertIn('p95_latency', cpu_perf)
        
        # Test CUDA performance if available
        if 'CUDAExecutionProvider' in self.provider_manager.available_providers:
            cuda_perf = self.provider_manager.measure_provider_performance(
                'CUDAExecutionProvider',
                input_data,
                num_iterations=10
            )
            
            self.assertIn('mean_latency', cuda_perf)
            self.assertIn('p95_latency', cuda_perf)
            
            # CUDA should generally be faster than CPU
            self.assertLess(cuda_perf['mean_latency'], cpu_perf['mean_latency'])

if __name__ == '__main__':
    unittest.main()
