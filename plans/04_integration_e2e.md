# Phase 4: Integration and End-to-End Tests

## Pipeline Integration Tests

### Training Pipeline
- Test end-to-end DSPy training
- Test model checkpoint saving
- Test training resumption
- Test multi-GPU training
- Test distributed training
- Test mixed precision training
- Test gradient accumulation
- Test learning rate scheduling
- Test early stopping
- Test validation integration

### Export Pipeline
- Test model format conversion
- Test weight transfer accuracy
- Test dynamic axes handling
- Test custom operator export
- Test graph optimization during export
- Test input/output validation
- Test shape inference
- Test metadata preservation
- Test version compatibility
- Test export error handling

### Inference Pipeline
- Test batch processing
- Test streaming inference
- Test async inference
- Test multi-threading
- Test memory management
- Test error recovery
- Test warmup procedures
- Test throughput optimization
- Test latency monitoring
- Test resource cleanup

### Optimization Pipeline
- Test quantization workflow
- Test provider selection
- Test graph optimization
- Test performance profiling
- Test memory profiling
- Test hardware utilization
- Test error handling
- Test optimization validation
- Test model comparison
- Test deployment preparation

## CLI Integration Tests

### Command Validation
- Test argument parsing
- Test configuration loading
- Test environment setup
- Test input validation
- Test path resolution
- Test default values
- Test required parameters
- Test parameter dependencies
- Test parameter conflicts
- Test help documentation

### Progress Reporting
- Test progress bar updates
- Test status messages
- Test error messages
- Test warning handling
- Test log levels
- Test output formatting
- Test color support
- Test interactive mode
- Test quiet mode
- Test verbose mode

### Result Handling
- Test success reporting
- Test error reporting
- Test performance metrics
- Test accuracy metrics
- Test resource usage stats
- Test output formatting
- Test result persistence
- Test comparison tools
- Test visualization
- Test export formats

## End-to-End Test Scenarios

### Basic Workflow
```python
class TestBasicWorkflow(unittest.TestCase):
    def setUp(self):
        self.cli = CommandLineInterface()
        self.test_data = self.prepare_test_data()
        self.config = self.load_test_config()
        
    def prepare_test_data(self):
        return {
            "train": create_synthetic_dataset(1000),
            "val": create_synthetic_dataset(100),
            "test": create_synthetic_dataset(100)
        }
        
    def load_test_config(self):
        return {
            "model_type": "classifier",
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 1e-4,
            "quantization": "dynamic",
            "provider": "CUDAExecutionProvider"
        }
    
    def test_complete_pipeline(self):
        # Train model
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.config
        })
        self.assertIsNotNone(model)
        
        # Export to ONNX
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": "model.onnx"
        })
        self.assertTrue(os.path.exists(onnx_path))
        
        # Optimize
        optimized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": self.config["quantization"],
            "provider": self.config["provider"]
        })
        self.assertTrue(os.path.exists(optimized_path))
        
        # Run inference
        results = self.cli.run_command("infer", {
            "model": optimized_path,
            "data": self.test_data["test"],
            "batch_size": self.config["batch_size"]
        })
        self.assertIsNotNone(results)
        
        # Validate results
        metrics = calculate_metrics(results, self.test_data["test"])
        self.assertGreater(metrics["accuracy"], 0.9)
```

### Advanced Scenarios
```python
class TestAdvancedScenarios(unittest.TestCase):
    def test_distributed_training(self):
        # Test multi-GPU training
        results = self.cli.run_command("train", {
            "distributed": True,
            "num_gpus": 2,
            **self.config
        })
        self.verify_distributed_training(results)
    
    def test_quantization_accuracy(self):
        # Test accuracy retention after quantization
        original_model = self.train_model()
        original_accuracy = self.evaluate_model(original_model)
        
        quantized_model = self.cli.run_command("optimize", {
            "model": original_model,
            "quantization": "static",
            "calibration_data": self.test_data["val"]
        })
        
        quantized_accuracy = self.evaluate_model(quantized_model)
        self.assertGreater(quantized_accuracy, original_accuracy * 0.99)
    
    def test_error_recovery(self):
        # Test pipeline recovery from failures
        with self.assertRaises(Exception):
            self.cli.run_command("train", {
                "invalid_param": True
            })
        
        # Should continue with valid parameters
        result = self.cli.run_command("train", self.config)
        self.assertIsNotNone(result)
```

### Performance Tests
```python
class TestPerformance(unittest.TestCase):
    @pytest.mark.benchmark
    def test_training_performance(self):
        # Measure training time and resource usage
        with ResourceMonitor() as monitor:
            model = self.cli.run_command("train", self.config)
        
        stats = monitor.get_stats()
        self.assertLess(stats["gpu_memory_peak"], 8 * 1024 * 1024 * 1024)  # 8GB
        self.assertLess(stats["training_time"], 3600)  # 1 hour
    
    @pytest.mark.benchmark
    def test_inference_performance(self):
        # Measure inference latency and throughput
        model = self.load_optimized_model()
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(100):
                futures.append(
                    executor.submit(
                        self.cli.run_command,
                        "infer",
                        {"model": model, "data": self.test_data["test"]}
                    )
                )
            
            latencies = [f.result()["latency"] for f in futures]
            
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        self.assertLess(avg_latency, 0.1)  # 100ms
        self.assertLess(p95_latency, 0.2)  # 200ms
```

## Success Criteria

### Integration Quality
- All pipeline stages connect correctly
- Data flows properly between stages
- Error handling works end-to-end
- Resource management is robust

### CLI Usability
- Commands are intuitive
- Help documentation is clear
- Error messages are helpful
- Progress reporting is accurate

### Performance Metrics
- Training completes in expected time
- Inference meets latency targets
- Resource usage within bounds
- Scaling behavior is efficient

### Reliability
- No memory leaks
- Proper error recovery
- Consistent results
- Stable under load

## Development Workflow
1. Implement basic pipeline
2. Add CLI interface
3. Add monitoring and metrics
4. Test error scenarios
5. Optimize performance
6. Document usage

## Tools and Dependencies
- pytest-integration
- pytest-benchmark
- pytest-timeout
- pytest-xdist for parallel testing
- memory_profiler
- psutil
- GPUtil
- tqdm
