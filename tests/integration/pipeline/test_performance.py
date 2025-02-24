"""Performance and benchmark tests."""

import os
import time
import unittest
import pytest
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestPerformance(BasePipelineTest):
    """Test performance characteristics and benchmarks."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        self.test_data = {
            "train": self.create_synthetic_dataset(1000),
            "val": self.create_synthetic_dataset(100),
            "test": self.create_synthetic_dataset(100)
        }
        
    @pytest.mark.benchmark
    def test_training_performance(self):
        """Test training time and resource usage."""
        with self.resource_monitor() as monitor:
            model = self.cli.run_command("train", {
                "data": self.test_data["train"],
                "val_data": self.test_data["val"],
                **self.default_config
            })
            
        stats = monitor.get_stats()
        
        # Verify training completed in reasonable time
        self.assertLess(stats["training_time"], 3600)  # 1 hour max
        
        # Check GPU memory usage if available
        if torch.cuda.is_available():
            self.assertLess(
                stats["gpu_memory_peak"],
                8 * 1024 * 1024 * 1024  # 8GB max
            )
            
        # Verify model quality
        results = self.cli.run_command("infer", {
            "model": model,
            "data": self.test_data["test"]
        })
        metrics = self.calculate_metrics(results, self.test_data["test"])
        self.assertGreater(metrics["accuracy"], 0.8)
        
    @pytest.mark.benchmark
    def test_inference_performance(self):
        """Test inference latency and throughput."""
        # Train and export model
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.default_config
        })
        
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": str(self.model_path)
        })
        
        optimized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": self.default_config["quantization"],
            "provider": self.default_config["provider"],
            "output": str(self.optimized_path)
        })
        
        # Test parallel inference
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(100):
                futures.append(
                    executor.submit(
                        self.cli.run_command,
                        "infer",
                        {
                            "model": optimized_path,
                            "data": self.test_data["test"],
                            "batch_size": self.default_config["batch_size"]
                        }
                    )
                )
            
            # Collect latencies
            latencies = []
            for future in futures:
                result = future.result()
                latencies.append(result["latency"])
                
        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Verify latency requirements
        self.assertLess(avg_latency, 0.1)  # 100ms average
        self.assertLess(p95_latency, 0.2)  # 200ms 95th percentile
        self.assertLess(p99_latency, 0.5)  # 500ms 99th percentile
        
    @pytest.mark.benchmark
    def test_memory_management(self):
        """Test memory usage and cleanup during operations."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Run full pipeline
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.default_config
        })
        
        training_memory = process.memory_info().rss
        
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": str(self.model_path)
        })
        
        export_memory = process.memory_info().rss
        
        optimized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": self.default_config["quantization"],
            "provider": self.default_config["provider"],
            "output": str(self.optimized_path)
        })
        
        optimize_memory = process.memory_info().rss
        
        # Run garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        final_memory = process.memory_info().rss
        
        # Verify reasonable memory usage
        self.assertLess(training_memory - initial_memory, 2 * 1024 * 1024 * 1024)  # 2GB max increase
        self.assertLess(final_memory - initial_memory, 512 * 1024 * 1024)  # 512MB residual
        
        # Verify no memory leaks during inference
        for _ in range(10):
            results = self.cli.run_command("infer", {
                "model": optimized_path,
                "data": self.test_data["test"]
            })
            gc.collect()
            torch.cuda.empty_cache()
            current_memory = process.memory_info().rss
            self.assertLess(current_memory - final_memory, 256 * 1024 * 1024)  # 256MB max increase
