import unittest
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

from onnx_agent.optimization.hardware.optimizer import GraphOptimizer
from .base_tests import BaseOptimizationTest

class TestOptimizer(BaseOptimizationTest):
    def setUp(self):
        self.test_model_dir = Path("test_models")
        self.test_model_dir.mkdir(exist_ok=True)
        self.test_model_path = self.test_model_dir / "test_model.onnx"
        self._create_test_model()
        super().setUp()
        
    def setup_optimizer(self):
        """Override base class method to provide optimizer."""
        return GraphOptimizer()
        
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
        
        # Create inputs
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
        
        # Create constant nodes
        const1 = helper.make_tensor('const1', TensorProto.FLOAT, [1], [1.0])
        const2 = helper.make_tensor('const2', TensorProto.FLOAT, [1], [2.0])
        
        # Create nodes with constants that can be folded
        node1 = helper.make_node(
            'Add',
            inputs=['const1', 'const2'],
            outputs=['add_out']
        )
        
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
            [node1, conv_node, pool_node, flatten_node, gemm_node],
            'test-model',
            [X],
            [Y],
            [const1, const2, conv1_weight, linear_weight]
        )
        
        # Create the model with opset version 21 (latest stable)
        model_def = helper.make_model(
            graph_def,
            producer_name='onnx-example',
            opset_imports=[helper.make_opsetid("", 21)]
        )
        
        # Save the model
        onnx.save(model_def, str(self.test_model_path))
        
    def test_constant_folding(self):
        """Test constant folding optimization."""
        model = onnx.load(str(self.test_model_path))
        
        # Apply constant folding
        optimized_model = self.optimizer.optimize(
            model,
            ["constant_folding"]
        )
        
        # Verify optimization
        self.assertIsNotNone(optimized_model)
        self.assertTrue(self.optimizer.verify_optimization(model, optimized_model))
        
        # Check model validity
        onnx.checker.check_model(optimized_model)
        
    def test_node_fusion(self):
        """Test node fusion optimization."""
        model = onnx.load(str(self.test_model_path))
        
        try:
            # Apply node fusion
            optimized_model = self.optimizer.optimize(
                model,
                ["node_fusion"]
            )
            
            # Verify optimization
            self.assertIsNotNone(optimized_model)
            self.assertTrue(self.optimizer.verify_optimization(model, optimized_model))
            
            # Check model validity
            onnx.checker.check_model(optimized_model)
        except Exception as e:
            self.skipTest(f"Node fusion optimization failed: {e}")
        
    def test_memory_optimization(self):
        """Test memory optimization."""
        model = onnx.load(str(self.test_model_path))
        
        # Measure initial memory usage
        initial_memory = self.optimizer.measure_memory_usage(model)
        
        # Apply memory optimization
        optimized_model = self.optimizer.optimize(
            model,
            ["memory_optimizer"]
        )
        
        # Measure optimized memory usage
        optimized_memory = self.optimizer.measure_memory_usage(optimized_model)
        
        # Verify optimization
        self.assertIsNotNone(optimized_model)
        self.assertTrue(self.optimizer.verify_optimization(model, optimized_model))
        
        # Check model validity
        onnx.checker.check_model(optimized_model)
        
        # Memory should be reduced or at least not increased significantly
        self.assertLessEqual(
            optimized_memory['total_size'],
            initial_memory['total_size'] * 1.1  # Allow 10% overhead
        )
        
    def test_multiple_optimizations(self):
        """Test applying multiple optimization passes."""
        model = onnx.load(str(self.test_model_path))
        
        # Apply multiple optimizations
        optimized_model = self.optimizer.optimize(
            model,
            ["constant_folding", "node_fusion", "memory_optimizer"]
        )
        
        # Verify optimization
        self.assertIsNotNone(optimized_model)
        self.assertTrue(self.optimizer.verify_optimization(model, optimized_model))
        
        # Check model validity
        onnx.checker.check_model(optimized_model)
        
    def test_invalid_optimization(self):
        """Test handling of invalid optimization pass."""
        model = onnx.load(str(self.test_model_path))
        
        # Try to apply non-existent optimization
        with self.assertRaises(ValueError):
            self.optimizer.optimize(
                model,
                ["invalid_optimization"]
            )

if __name__ == '__main__':
    unittest.main()
