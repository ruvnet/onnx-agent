import unittest
import numpy as np
import torch
import torch.nn as nn
from src.onnx_agent.test_time_compute.ttt import TTTOptimizer
from .base_tests import BaseTTTTest

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*224*224, 10)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)

class TestTTT(BaseTTTTest):
    def setUp(self):
        super().setUp()
        self.model = SimpleModel()
        self.ttt_optimizer = TTTOptimizer()
        self.ttt_optimizer.setup(self.model, learning_rate=1e-4)
        
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.ttt_optimizer.model)
        self.assertIsNotNone(self.ttt_optimizer.optimizer)
        self.assertIsNotNone(self.ttt_optimizer.scheduler)
        self.assertIsNotNone(self.ttt_optimizer.initial_state)
        
    def test_adaptation_step(self):
        """Test single adaptation step"""
        test_batch = torch.from_numpy(self.test_batch)
        initial_loss = self.ttt_optimizer.compute_loss(test_batch)
        loss = self.ttt_optimizer.adaptation_step(test_batch)
        self.assertIsInstance(loss, float)
        
    def test_convergence(self):
        """Test adaptation convergence"""
        test_batch = torch.from_numpy(self.test_batch)
        history = self.ttt_optimizer.adapt(
            test_batch, 
            self.adaptation_config
        )
        self.assertIn("converged", history)
        self.assertIn("losses", history)
        self.assertIn("iterations", history)
        self.assertGreater(len(history["losses"]), 0)
        
    def test_reset(self):
        """Test model reset to initial state"""
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Run adaptation
        test_batch = torch.from_numpy(self.test_batch)
        self.ttt_optimizer.adapt(test_batch, self.adaptation_config)
        
        # Reset model
        self.ttt_optimizer.reset()
        
        # Verify parameters are reset
        for name, param in self.model.named_parameters():
            self.assertTrue(
                torch.allclose(param, initial_params[name])
            )
            
    def test_uninitialized_error(self):
        """Test error handling for uninitialized model"""
        optimizer = TTTOptimizer()
        test_batch = torch.from_numpy(self.test_batch)
        
        with self.assertRaises(ValueError):
            optimizer.compute_loss(test_batch)
            
        with self.assertRaises(ValueError):
            optimizer.adaptation_step(test_batch)
            
        with self.assertRaises(ValueError):
            optimizer.reset()
            
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        test_batch = torch.from_numpy(self.test_batch)
        
        # Run adaptation with enough iterations
        config = {
            "max_iterations": 10,
            "convergence_threshold": 1e-10  # Small threshold to force iterations
        }
        history = self.ttt_optimizer.adapt(test_batch, config)
        
        # Verify loss decreases over time
        initial_loss = history["initial_loss"]
        final_loss = history["final_loss"]
        self.assertGreater(initial_loss, final_loss)
        self.assertGreater(len(history["losses"]), 1)
