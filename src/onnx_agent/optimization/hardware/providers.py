from typing import Dict, List, Optional, Union
import numpy as np
import onnxruntime as ort

class ExecutionProviderManager:
    """Manages execution providers for ONNX Runtime inference."""
    
    def __init__(self, model_path: str):
        """Initialize provider manager.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.available_providers = ort.get_available_providers()
        
    def create_session_with_provider(self, 
                                   provider: str,
                                   provider_options: Optional[Dict] = None,
                                   session_options: Optional[ort.SessionOptions] = None) -> ort.InferenceSession:
        """Create inference session with specified provider.
        
        Args:
            provider: Name of execution provider
            provider_options: Provider-specific options
            session_options: General session options
            
        Returns:
            ort.InferenceSession: Configured session
        """
        if provider not in self.available_providers:
            raise ValueError(f"Provider {provider} not available")
            
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
        providers = [(provider, provider_options or {})]
        
        # Add CPU as fallback
        if provider != 'CPUExecutionProvider':
            providers.append('CPUExecutionProvider')
            
        return ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )
        
    def configure_cuda_provider(self, 
                              device_id: int = 0,
                              memory_limit: Optional[int] = None,
                              arena_extend_strategy: str = 'kNextPowerOfTwo',
                              gpu_mem_limit: Optional[int] = None,
                              cudnn_conv_algo_search: str = 'EXHAUSTIVE',
                              do_copy_in_default_stream: bool = True) -> Dict:
        """Configure CUDA execution provider options.
        
        Args:
            device_id: CUDA device ID
            memory_limit: GPU memory limit in bytes
            arena_extend_strategy: Memory arena extension strategy
            gpu_mem_limit: GPU memory limit for caching
            cudnn_conv_algo_search: cuDNN convolution algorithm search strategy
            do_copy_in_default_stream: Use default CUDA stream for copies
            
        Returns:
            Dict of CUDA provider options
        """
        return {
            'device_id': device_id,
            'gpu_mem_limit': gpu_mem_limit,
            'arena_extend_strategy': arena_extend_strategy,
            'cudnn_conv_algo_search': cudnn_conv_algo_search,
            'do_copy_in_default_stream': do_copy_in_default_stream,
            **({"memory_limit": memory_limit} if memory_limit else {})
        }
        
    def configure_tensorrt_provider(self,
                                  device_id: int = 0,
                                  trt_max_workspace_size: int = 2147483648,
                                  trt_fp16_enable: bool = False,
                                  trt_int8_enable: bool = False,
                                  trt_engine_cache_enable: bool = True,
                                  trt_engine_cache_path: str = "") -> Dict:
        """Configure TensorRT execution provider options.
        
        Args:
            device_id: GPU device ID
            trt_max_workspace_size: Maximum workspace size
            trt_fp16_enable: Enable FP16 precision
            trt_int8_enable: Enable INT8 precision
            trt_engine_cache_enable: Enable engine caching
            trt_engine_cache_path: Path to cache TRT engines
            
        Returns:
            Dict of TensorRT provider options
        """
        return {
            'device_id': device_id,
            'trt_max_workspace_size': trt_max_workspace_size,
            'trt_fp16_enable': trt_fp16_enable,
            'trt_int8_enable': trt_int8_enable,
            'trt_engine_cache_enable': trt_engine_cache_enable,
            'trt_engine_cache_path': trt_engine_cache_path
        }
        
    def verify_provider_compatibility(self, model_path: str, provider: str) -> bool:
        """Verify if a model is compatible with an execution provider.
        
        Args:
            model_path: Path to ONNX model
            provider: Name of execution provider
            
        Returns:
            bool: True if compatible
        """
        try:
            session = self.create_session_with_provider(provider)
            # Run a small inference to verify
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            session.run(None, {"input": dummy_input})
            return True
        except Exception as e:
            print(f"Provider compatibility check failed: {e}")
            return False
            
    def measure_provider_performance(self,
                                   provider: str,
                                   input_data: np.ndarray,
                                   num_iterations: int = 100) -> Dict[str, float]:
        """Measure inference performance with specified provider.
        
        Args:
            provider: Name of execution provider
            input_data: Input data for inference
            num_iterations: Number of iterations for measurement
            
        Returns:
            Dict with latency statistics
        """
        import time
        
        session = self.create_session_with_provider(provider)
        latencies = []
        
        # Warmup
        for _ in range(10):
            session.run(None, {"input": input_data})
            
        # Measure
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {"input": input_data})
            latencies.append(time.perf_counter() - start)
            
        return {
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        }
