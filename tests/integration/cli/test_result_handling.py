"""CLI result handling tests."""

import os
import json
import unittest
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestResultHandling(BasePipelineTest):
    """Test CLI result handling and reporting."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        self.test_data = self.create_synthetic_dataset(100)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test artifacts."""
        super().tearDown()
        if self.results_dir.exists():
            for file in self.results_dir.glob("*"):
                file.unlink()
            self.results_dir.rmdir()
            
    def test_success_reporting(self):
        """Test successful operation reporting."""
        # Test training success
        result = self.cli.run_command("train", {
            "data": self.test_data,
            **self.default_config
        })
        
        self.assertIsNotNone(result)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        
        # Test export success
        export_result = self.cli.run_command("export", {
            "model": result["model"],
            "output": str(self.model_path)
        })
        
        self.assertIsNotNone(export_result)
        self.assertEqual(export_result["status"], "success")
        self.assertTrue(self.model_path.exists())
        
    def test_error_reporting(self):
        """Test error reporting and handling."""
        # Test invalid model error
        with self.assertRaises(Exception) as ctx:
            self.cli.run_command("infer", {
                "model": "nonexistent.onnx",
                "data": self.test_data
            })
            
        error = ctx.exception
        self.assertIsInstance(error, Exception)
        self.assertIn("error_code", error.__dict__)
        self.assertIn("details", error.__dict__)
        
        # Test validation error
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("train", {
                "data": self.test_data,
                "batch_size": -1
            })
            
        self.assertIn("batch_size", str(ctx.exception))
        
    def test_performance_metrics(self):
        """Test performance metric reporting."""
        # Train and optimize model
        model = self.cli.run_command("train", {
            "data": self.test_data,
            **self.default_config
        })["model"]
        
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": str(self.model_path)
        })["path"]
        
        # Run inference with metrics
        result = self.cli.run_command("infer", {
            "model": onnx_path,
            "data": self.test_data,
            "collect_metrics": True
        })
        
        # Verify metrics
        self.assertIn("metrics", result)
        metrics = result["metrics"]
        self.assertIn("latency", metrics)
        self.assertIn("throughput", metrics)
        self.assertIn("memory_usage", metrics)
        
        # Verify metric values
        self.assertGreater(metrics["throughput"], 0)
        self.assertGreater(metrics["latency"], 0)
        self.assertGreater(metrics["memory_usage"], 0)
        
    def test_accuracy_metrics(self):
        """Test accuracy metric reporting."""
        # Train model
        model = self.cli.run_command("train", {
            "data": self.test_data,
            **self.default_config
        })["model"]
        
        # Test with validation data
        result = self.cli.run_command("evaluate", {
            "model": model,
            "data": self.test_data
        })
        
        # Verify metrics
        self.assertIn("metrics", result)
        metrics = result["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("loss", metrics)
        self.assertGreater(metrics["accuracy"], 0)
        
    def test_resource_usage_stats(self):
        """Test resource usage statistics."""
        # Run training with resource monitoring
        result = self.cli.run_command("train", {
            "data": self.test_data,
            "monitor_resources": True,
            **self.default_config
        })
        
        # Verify resource stats
        self.assertIn("resources", result)
        resources = result["resources"]
        self.assertIn("cpu_usage", resources)
        self.assertIn("memory_usage", resources)
        self.assertIn("gpu_usage", resources)
        
        # Verify reasonable values
        self.assertGreater(resources["cpu_usage"], 0)
        self.assertGreater(resources["memory_usage"], 0)
        
    def test_result_persistence(self):
        """Test result saving and loading."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"results_{timestamp}.json"
        
        # Run inference and save results
        result = self.cli.run_command("infer", {
            "model": "model.onnx",
            "data": self.test_data,
            "save_results": str(results_file)
        })
        
        # Verify file exists and contains results
        self.assertTrue(results_file.exists())
        with open(results_file) as f:
            saved_results = json.load(f)
        self.assertEqual(saved_results["status"], result["status"])
        
    def test_comparison_tools(self):
        """Test result comparison functionality."""
        # Train two models
        model1 = self.cli.run_command("train", {
            "data": self.test_data,
            **self.default_config
        })["model"]
        
        model2 = self.cli.run_command("train", {
            "data": self.test_data,
            **self.default_config
        })["model"]
        
        # Compare models
        comparison = self.cli.run_command("compare", {
            "models": [model1, model2],
            "data": self.test_data
        })
        
        # Verify comparison results
        self.assertIn("comparison", comparison)
        self.assertIn("metrics", comparison["comparison"])
        self.assertIn("differences", comparison["comparison"])
        
    def test_visualization(self):
        """Test result visualization."""
        # Generate visualizations
        result = self.cli.run_command("train", {
            "data": self.test_data,
            "visualize": True,
            "visualization_path": str(self.results_dir),
            **self.default_config
        })
        
        # Verify visualization files
        self.assertTrue((self.results_dir / "training_progress.png").exists())
        self.assertTrue((self.results_dir / "loss_curve.png").exists())
        self.assertTrue((self.results_dir / "accuracy_curve.png").exists())
