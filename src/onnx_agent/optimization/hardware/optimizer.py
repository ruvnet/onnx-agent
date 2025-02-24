from typing import Dict, List, Optional, Union
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort

class GraphOptimizer:
    """Handles ONNX graph optimizations."""
    
    def __init__(self):
        """Initialize graph optimizer."""
        self.optimization_passes = {
            "constant_folding": self._apply_constant_folding,
            "node_fusion": self._apply_node_fusion,
            "memory_optimizer": self._apply_memory_optimization
        }
        
    def optimize(self, 
                model: onnx.ModelProto,
                passes: List[str],
                **kwargs) -> onnx.ModelProto:
        """Apply optimization passes to the model.
        
        Args:
            model: ONNX model to optimize
            passes: List of optimization passes to apply
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized ONNX model
        """
        optimized_model = model
        for pass_name in passes:
            if pass_name in self.optimization_passes:
                optimized_model = self.optimization_passes[pass_name](optimized_model, **kwargs)
            else:
                raise ValueError(f"Unknown optimization pass: {pass_name}")
                
        return optimized_model
        
    def _apply_constant_folding(self, 
                              model: onnx.ModelProto,
                              **kwargs) -> onnx.ModelProto:
        """Apply constant folding optimization.
        
        Args:
            model: Input model
            **kwargs: Additional parameters
            
        Returns:
            Optimized model
        """
        try:
            import onnxoptimizer
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes',
                'fuse_constants',
                'eliminate_unused_initializer'
            ]
            opt_model = onnxoptimizer.optimize(model, passes)
            return opt_model
        except ModuleNotFoundError:
            print("onnxoptimizer not available, skipping constant folding optimization")
            return model
        
    def _apply_node_fusion(self,
                          model: onnx.ModelProto,
                          **kwargs) -> onnx.ModelProto:
        """Apply node fusion optimization.
        
        Args:
            model: Input model
            **kwargs: Additional parameters
            
        Returns:
            Optimized model
        """
        try:
            import onnxoptimizer
            passes = [
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_matmul_add_bias_into_gemm'
            ]
            opt_model = onnxoptimizer.optimize(model, passes)
            return opt_model
        except ModuleNotFoundError:
            print("onnxoptimizer not available, skipping node fusion optimization")
            return model
        
    def _apply_memory_optimization(self,
                                 model: onnx.ModelProto,
                                 **kwargs) -> onnx.ModelProto:
        """Apply memory usage optimization.
        
        Args:
            model: Input model
            **kwargs: Additional parameters
            
        Returns:
            Optimized model
        """
        import tempfile
        
        # Configure session for memory optimization
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Save optimized model
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            sess_options.optimized_model_filepath = tmp.name
            _ = ort.InferenceSession(
                model.SerializeToString(),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            optimized_model = onnx.load(tmp.name)
            
        return optimized_model
        
    def verify_optimization(self, 
                          original_model: onnx.ModelProto,
                          optimized_model: onnx.ModelProto) -> bool:
        """Verify optimization results.
        
        Args:
            original_model: Original ONNX model
            optimized_model: Optimized ONNX model
            
        Returns:
            bool: True if verification passes
        """
        try:
            # Check model validity
            onnx.checker.check_model(optimized_model)
            
            # Verify node count reduction
            original_nodes = len(original_model.graph.node)
            optimized_nodes = len(optimized_model.graph.node)
            
            # Basic verification - should have fewer or equal nodes
            if optimized_nodes > original_nodes:
                print(f"Warning: Optimized model has more nodes ({optimized_nodes}) than original ({original_nodes})")
                
            return True
        except Exception as e:
            print(f"Optimization verification failed: {e}")
            return False
            
    def measure_memory_usage(self, model: onnx.ModelProto) -> Dict[str, int]:
        """Measure model memory usage.
        
        Args:
            model: ONNX model to measure
            
        Returns:
            Dict with memory statistics
        """
        import sys
        
        def get_size(obj) -> int:
            """Get size of object and its members in bytes."""
            seen = set()
            
            def sizeof(obj):
                if id(obj) in seen:
                    return 0
                seen.add(id(obj))
                size = sys.getsizeof(obj)
                
                if hasattr(obj, '__dict__'):
                    size += sum(sizeof(v) for v in obj.__dict__.values())
                elif hasattr(obj, '__slots__'):
                    size += sum(sizeof(getattr(obj, s)) for s in obj.__slots__)
                elif isinstance(obj, dict):
                    size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
                elif isinstance(obj, (list, tuple, set, frozenset)):
                    size += sum(sizeof(i) for i in obj)
                    
                return size
                
            return sizeof(obj)
            
        total_size = get_size(model)
        graph_size = get_size(model.graph)
        weights_size = sum(get_size(t) for t in model.graph.initializer)
        
        return {
            "total_size": total_size,
            "graph_size": graph_size,
            "weights_size": weights_size,
            "other_size": total_size - graph_size - weights_size
        }
