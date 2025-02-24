"""Pytest configuration for integration tests."""

import os
import pytest
import numpy as np
import torch
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """Return the test directory path."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def data_dir(test_dir):
    """Create and return a temporary data directory for test artifacts."""
    data_path = test_dir / "data"
    data_path.mkdir(exist_ok=True)
    return data_path

@pytest.fixture
def create_synthetic_dataset():
    """Create a synthetic dataset for testing."""
    def _create_dataset(size, features=10, classes=2):
        X = np.random.randn(size, features).astype(np.float32)
        y = np.random.randint(0, classes, size=size).astype(np.int64)
        return {"features": torch.from_numpy(X), "labels": torch.from_numpy(y)}
    return _create_dataset

@pytest.fixture
def calculate_metrics():
    """Calculate metrics for model evaluation."""
    def _calculate_metrics(predictions, targets):
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
    return _calculate_metrics

@pytest.fixture
def resource_monitor():
    """Monitor system resources during test execution."""
    class ResourceMonitor:
        def __init__(self):
            self.stats = {}
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stats["training_time"] = time.time() - self.start_time
            if torch.cuda.is_available():
                self.stats["gpu_memory_peak"] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            
        def get_stats(self):
            return self.stats
            
    return ResourceMonitor

@pytest.fixture
def cli():
    """Create a CommandLineInterface instance for testing."""
    from onnx_agent.cli import CommandLineInterface
    return CommandLineInterface()
