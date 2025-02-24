"""Base classes and utilities for pipeline integration tests."""

import os
import unittest
import pytest
import torch
import numpy as np
from pathlib import Path

class BasePipelineTest(unittest.TestCase):
    """Base class for pipeline integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path(__file__).parent
        cls.data_dir = cls.test_dir / "data"
        cls.data_dir.mkdir(exist_ok=True)
        
        # Default configuration
        cls.default_config = {
            "model_type": "classifier",
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 1e-4,
            "quantization": "dynamic",
            "provider": "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        }
        
    def setUp(self):
        """Set up test case."""
        self.model_path = self.data_dir / "test_model.onnx"
        self.optimized_path = self.data_dir / "optimized_model.onnx"
        
    def tearDown(self):
        """Clean up test artifacts."""
        if self.model_path.exists():
            self.model_path.unlink()
        if self.optimized_path.exists():
            self.optimized_path.unlink()
            
    def create_synthetic_dataset(self, size, features=10, classes=2):
        """Create synthetic dataset for testing."""
        X = np.random.randn(size, features).astype(np.float32)
        y = np.random.randint(0, classes, size=size).astype(np.int64)
        return {
            "features": torch.from_numpy(X),
            "labels": torch.from_numpy(y)
        }
        
    def calculate_metrics(self, predictions, targets):
        """Calculate evaluation metrics."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, dict):
            targets = targets["labels"].numpy()
        elif isinstance(targets, torch.Tensor):
            targets = targets.numpy()
            
        correct = (predictions.argmax(axis=1) == targets).sum()
        total = len(targets)
        return {
            "accuracy": correct / total,
            "samples": total
        }
        
    def verify_model_exists(self, path):
        """Verify model file exists and is valid."""
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        
    def verify_model_outputs(self, outputs, expected_shape):
        """Verify model output format and shape."""
        self.assertIsNotNone(outputs)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        self.assertEqual(outputs.shape, expected_shape)
