import unittest
import numpy as np
import torch
import torch.nn as nn
from src.onnx_agent.test_time_compute.iterative import IterativeInference
from .base_tests import BaseIterativeTest

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*224*224, 10)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)

class TestIterativeInference(BaseIterativeTest):
    def setUp(self):
        super().setUp()
        self.model = SimpleModel()
        self.iterator = IterativeInference()
        self.iterator.setup(self.model)
        
    def test_initialization(self):
        """Test iterator initialization"""
        self.assertIsNotNone(self.iterator.model)
        
    def test_confidence_computation(self):
        """Test confidence score computation"""
        prediction = np.random.rand(1, 10)
        confidence = self.iterator.compute_confidence(prediction)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_stability_check(self):
        """Test prediction stability checking"""
        # Create stable predictions
        predictions = [
            np.array([[0.9, 0.1], [0.8, 0.2]]) for _ in range(5)
        ]
        self.assertTrue(
            self.iterator.check_stability(predictions, window_size=3)
        )
        
        # Create unstable predictions
        unstable = [
            np.array([[0.1, 0.9], [0.2, 0.8]]),
            np.array([[0.9, 0.1], [0.8, 0.2]])
        ]
        self.assertFalse(
            self.iterator.check_stability(unstable, window_size=2)
        )
        
    def test_basic_iteration(self):
        """Test basic iteration loop"""
        test_sample = torch.from_numpy(self.test_sample)
        result = self.iterator.run_inference(
            test_sample, 
            self.iteration_config
        )
        self.assertIsNotNone(result.final_prediction)
        self.assertIsInstance(result.confidence, float)
        self.assertGreater(result.num_iterations, 0)
        self.assertIsInstance(result.converged, bool)
        
    def test_early_stopping_confidence(self):
        """Test early stopping based on confidence"""
        config = {
            **self.iteration_config,
            "confidence_threshold": 0.1  # Low threshold to trigger early stop
        }
        test_sample = torch.from_numpy(self.test_sample)
        result = self.iterator.run_inference(test_sample, config)
        self.assertLess(
            result.num_iterations,
            config["max_iterations"]
        )
        
    def test_early_stopping_stability(self):
        """Test early stopping based on stability"""
        config = {
            **self.iteration_config,
            "stability_window": 2,
            "confidence_threshold": 1.0  # High threshold to force stability check
        }
        test_sample = torch.from_numpy(self.test_sample)
        result = self.iterator.run_inference(test_sample, config)
        self.assertLess(
            result.num_iterations,
            config["max_iterations"]
        )
        
    def test_max_iterations(self):
        """Test reaching max iterations"""
        config = {
            "max_iterations": 5,
            "confidence_threshold": 1.0,  # Impossible to reach
            "stability_window": 10  # Too large to trigger
        }
        test_sample = torch.from_numpy(self.test_sample)
        result = self.iterator.run_inference(test_sample, config)
        self.assertEqual(result.num_iterations, config["max_iterations"])
        self.assertFalse(result.converged)
        
    def test_uninitialized_error(self):
        """Test error handling for uninitialized model"""
        iterator = IterativeInference()
        test_sample = torch.from_numpy(self.test_sample)
        
        with self.assertRaises(ValueError):
            iterator.run_inference(test_sample, self.iteration_config)
