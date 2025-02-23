import unittest
import numpy as np

class BaseTTATest(unittest.TestCase):
    def setUp(self):
        self.tta_manager = None  # Will be initialized in implementation
        self.test_input = self.create_test_input()
        self.augmentations = self.setup_augmentations()
        
    def create_test_input(self):
        return np.random.randn(1, 3, 224, 224).astype(np.float32)
        
    def setup_augmentations(self):
        return ["horizontal_flip", "vertical_flip", "rotate90"]

class BaseTTTTest(unittest.TestCase):
    def setUp(self):
        self.ttt_optimizer = None  # Will be initialized in implementation
        self.test_batch = self.create_test_batch()
        self.adaptation_config = self.get_adaptation_config()
        
    def create_test_batch(self):
        return np.random.randn(32, 3, 224, 224).astype(np.float32)
        
    def get_adaptation_config(self):
        return {
            "learning_rate": 1e-4,
            "max_iterations": 10,
            "convergence_threshold": 1e-3
        }

class BaseIterativeTest(unittest.TestCase):
    def setUp(self):
        self.iterator = None  # Will be initialized in implementation
        self.test_sample = self.create_test_sample()
        self.iteration_config = self.get_iteration_config()
        
    def create_test_sample(self):
        return np.random.randn(1, 3, 224, 224).astype(np.float32)
        
    def get_iteration_config(self):
        return {
            "max_iterations": 20,
            "confidence_threshold": 0.95,
            "stability_window": 3
        }
