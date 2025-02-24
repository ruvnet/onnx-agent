"""Basic workflow integration tests."""

import os
import unittest
import pytest
import torch
import numpy as np
from pathlib import Path

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestBasicWorkflow(BasePipelineTest):
    """Test basic end-to-end workflow."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        self.test_data = {
            "train": self.create_synthetic_dataset(1000),
            "val": self.create_synthetic_dataset(100),
            "test": self.create_synthetic_dataset(100)
        }
        
    def test_complete_pipeline(self):
        """Test complete training, export, optimization, and inference pipeline."""
        # Train model
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            **self.default_config
        })
        self.assertIsNotNone(model)
        
        # Export to ONNX
        onnx_path = self.cli.run_command("export", {
            "model": model,
            "output": str(self.model_path)
        })
        self.verify_model_exists(Path(onnx_path))
        
        # Optimize
        optimized_path = self.cli.run_command("optimize", {
            "model": onnx_path,
            "quantization": self.default_config["quantization"],
            "provider": self.default_config["provider"],
            "output": str(self.optimized_path)
        })
        self.verify_model_exists(Path(optimized_path))
        
        # Run inference
        results = self.cli.run_command("infer", {
            "model": optimized_path,
            "data": self.test_data["test"],
            "batch_size": self.default_config["batch_size"]
        })
        self.assertIsNotNone(results)
        
        # Validate results
        metrics = self.calculate_metrics(results, self.test_data["test"])
        self.assertGreater(metrics["accuracy"], 0.8)
        
    def test_model_checkpoint_saving(self):
        """Test model checkpoint saving and loading."""
        checkpoint_path = self.data_dir / "checkpoint.pt"
        
        # Train with checkpoint saving
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            "checkpoint_path": str(checkpoint_path),
            **self.default_config
        })
        self.assertIsNotNone(model)
        self.assertTrue(checkpoint_path.exists())
        
        # Load from checkpoint and continue training
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            "checkpoint_path": str(checkpoint_path),
            "resume": True,
            **self.default_config
        })
        self.assertIsNotNone(model)
        
        # Clean up
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Train with early stopping
        model = self.cli.run_command("train", {
            "data": self.test_data["train"],
            "val_data": self.test_data["val"],
            "early_stopping": True,
            "patience": 2,
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
