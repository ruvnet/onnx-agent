"""Advanced integration test scenarios."""

import os
import unittest
import pytest
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestAdvancedScenarios(BasePipelineTest):
    """Test advanced scenarios and edge cases."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        self.test_data = {
            "train": self.create_synthetic_dataset(1000),
            "val": self.create_synthetic_dataset(100),
            "test": self.create_synthetic_dataset(100)
        }
        
    def test_distributed_training(self):
        """Test multi-GPU distributed training."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 GPUs")
            
        # Train with distributed setup
        results = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            "distributed": True,
            "num_gpus": 2,
            **self.default_config
        })
        self.verify_distributed_training(results)
        
    def verify_distributed_training(self, results):
        """Verify distributed training results."""
        self.assertIsNotNone(results)
        self.assertIn("model", results)
        self.assertIn("metrics", results)
        
        # Verify model performance
        test_results = self.cli.run_command("infer", {
            "model": results["model"],
            "data": self.test_data["test"]
        })
        metrics = self.calculate_metrics(test_results, self.test_data["test"])
        self.assertGreater(metrics["accuracy"], 0.8)
        
    def test_quantization_accuracy(self):
        """Test accuracy retention after quantization."""
        # Train original model
        original_model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.default_config
        })
        
        # Get original accuracy
        original_results = self.cli.run_command("infer", {
            "model": original_model,
            "data": self.test_data["test"]
        })
        original_metrics = self.calculate_metrics(original_results, self.test_data["test"])
        
        # Export to ONNX
        onnx_path = self.cli.run_command("export", {
            "model": original_model,
            "output": str(self.model_path)
        })
        
        # Quantize model
        quantized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": "static",
            "calibration_data": self.test_data["val"],
            "output": str(self.optimized_path)
        })
        
        # Test quantized model
        quantized_results = self.cli.run_command("infer", {
            "model": quantized_path,
            "data": self.test_data["test"]
        })
        quantized_metrics = self.calculate_metrics(quantized_results, self.test_data["test"])
        
        # Verify accuracy retention
        self.assertGreater(
            quantized_metrics["accuracy"],
            original_metrics["accuracy"] * 0.95  # Allow 5% accuracy drop
        )
        
    def test_error_recovery(self):
        """Test pipeline recovery from failures."""
        # Test invalid parameter handling
        with self.assertRaises(Exception):
            self.cli.run_command("train", {
                "invalid_param": True
            })
        
        # Verify can continue with valid parameters
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.default_config
        })
        self.assertIsNotNone(model)
        
        # Test recovery from optimization failure
        with self.assertRaises(Exception):
            self.cli.run_command("optimize", {
                "model": "nonexistent.onnx",
                "quantization": "invalid_mode"
            })
            
        # Verify can continue with valid optimization
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": str(self.model_path)
        })
        optimized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": "dynamic",
            "output": str(self.optimized_path)
        })
        self.verify_model_exists(Path(optimized_path))
        
    def test_mixed_precision_training(self):
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            self.skipTest("Requires GPU support")
            
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            "mixed_precision": True,
            **self.default_config
        })
        self.assertIsNotNone(model)
        
        # Verify model performance
        results = self.cli.run_command("infer", {
            "model": model,
            "data": self.test_data["test"]
        })
        metrics = self.calculate_metrics(results, self.test_data["test"])
        self.assertGreater(metrics["accuracy"], 0.8)
