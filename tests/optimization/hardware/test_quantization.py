import unittest
import numpy as np
import onnx
import os
from pathlib import Path

from onnx_agent.optimization.hardware.quantization import ONNXQuantizer
from .base_tests import BaseQuantizationTest

class TestQuantization(BaseQuantizationTest):
    def setUp(self):
        self.test_model_dir = Path("test_models")
        self.test_model_dir.mkdir(exist_ok=True)
        self.test_model_path = self.test_model_dir / "test_model.onnx"
        self._create_test_model()
        super().setUp()
        
    def setup_quantizer(self):
        """Override base class method to provide quantizer."""
        return ONNXQuantizer(str(self.test_model_path))
        
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
        
    def test_dynamic_quantization(self):
        """Test dynamic quantization process."""
        quantizer = ONNXQuantizer(str(self.test_model_path))
        
        # Perform dynamic quantization
        quantized_model = quantizer.quantize_dynamic(weight_type="QInt8")
        
        # Verify quantization
        self.assertIsNotNone(quantized_model)
        self.assertTrue(quantizer.verify_quantization(quantized_model))
        
        # Check model validity
        onnx.checker.check_model(quantized_model)
        
    def test_static_quantization(self):
        """Test static quantization with calibration."""
        quantizer = ONNXQuantizer(str(self.test_model_path))
        
        # Create calibration data
        calibration_data = np.random.randn(10, 3, 224, 224).astype(np.float32)
        
        # Perform static quantization
        quantized_model = quantizer.quantize_static(
            calibration_data,
            calibration_method="minmax"
        )
        
        # Verify quantization
        self.assertIsNotNone(quantized_model)
        self.assertTrue(quantizer.verify_quantization(quantized_model))
        
        # Check model validity
        onnx.checker.check_model(quantized_model)
        
    def test_accuracy_retention(self):
        """Test accuracy impact of quantization."""
        quantizer = ONNXQuantizer(str(self.test_model_path))
        
        # Measure original accuracy
        original_accuracy = quantizer.measure_accuracy(str(self.test_model_path))
        
        # Perform quantization
        quantized_model = quantizer.quantize_dynamic()
        
        # Measure quantized accuracy
        quantized_accuracy = quantizer.measure_accuracy(quantized_model)
        
        # Check accuracy retention (allowing for some degradation)
        self.assertGreater(quantized_accuracy, original_accuracy * 0.95)
        
    def test_model_size_reduction(self):
        """Test model size reduction after quantization."""
        import os
        
        quantizer = ONNXQuantizer(str(self.test_model_path))
        
        # Get original model size
        original_size = os.path.getsize(str(self.test_model_path))
        
        # Perform quantization
        quantized_model = quantizer.quantize_dynamic()
        
        # Save quantized model temporarily
        quantized_path = str(self.test_model_dir / "quantized_model.onnx")
        onnx.save(quantized_model, quantized_path)
        
        # Get quantized model size
        quantized_size = os.path.getsize(quantized_path)
        
        # Check size reduction (expecting at least 50% reduction)
        self.assertLess(quantized_size, original_size * 0.5)
        
        # Cleanup
        os.remove(quantized_path)
        
    def test_unsupported_ops_fallback(self):
        """Test fallback behavior for unsupported operators."""
        # This would require a more complex model with unsupported ops
        # For now, we'll just verify the basic quantization doesn't fail
        quantizer = ONNXQuantizer(str(self.test_model_path))
        
        try:
            quantized_model = quantizer.quantize_dynamic()
            self.assertIsNotNone(quantized_model)
        except Exception as e:
            self.fail(f"Quantization failed with unsupported ops: {e}")

if __name__ == '__main__':
    unittest.main()
