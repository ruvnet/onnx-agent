import unittest
import numpy as np
from src.onnx_agent.test_time_compute.tta import TTAManager
from .base_tests import BaseTTATest

class TestTTA(BaseTTATest):
    def setUp(self):
        super().setUp()
        self.tta_manager = TTAManager()

    def test_single_augmentation(self):
        """Test single augmentation application"""
        augmented = self.tta_manager.apply_augmentation(
            self.test_input, 
            "horizontal_flip"
        )
        self.assertEqual(augmented.shape, self.test_input.shape)
        # Verify flip by checking if arrays are different
        self.assertFalse(np.array_equal(augmented, self.test_input))
        
    def test_multiple_augmentations(self):
        """Test multiple augmentations"""
        results = self.tta_manager.apply_all_augmentations(
            self.test_input, 
            self.augmentations
        )
        self.assertEqual(len(results), len(self.augmentations))
        for result in results:
            self.assertEqual(result.shape, self.test_input.shape)
            
    def test_result_aggregation_mean(self):
        """Test mean aggregation of augmented results"""
        predictions = [np.random.rand(1, 1000) for _ in range(5)]
        aggregated = self.tta_manager.aggregate_results(predictions, "mean")
        self.assertEqual(aggregated.shape, (1, 1000))
        
    def test_result_aggregation_voting(self):
        """Test voting aggregation of augmented results"""
        predictions = [np.random.rand(1, 10) for _ in range(5)]
        aggregated = self.tta_manager.aggregate_results(predictions, "voting")
        self.assertIsInstance(aggregated, np.int64)
        
    def test_custom_augmentation(self):
        """Test custom augmentation registration"""
        def custom_aug(x):
            return x * 2
            
        self.tta_manager.register_augmentation("custom", custom_aug)
        augmented = self.tta_manager.apply_augmentation(
            self.test_input, 
            "custom"
        )
        self.assertTrue(np.array_equal(augmented, self.test_input * 2))
        
    def test_invalid_augmentation(self):
        """Test invalid augmentation handling"""
        with self.assertRaises(ValueError):
            self.tta_manager.apply_augmentation(
                self.test_input, 
                "nonexistent_aug"
            )
            
    def test_empty_predictions(self):
        """Test empty predictions list handling"""
        with self.assertRaises(ValueError):
            self.tta_manager.aggregate_results([])
            
    def test_invalid_aggregation_method(self):
        """Test invalid aggregation method handling"""
        predictions = [np.random.rand(1, 1000) for _ in range(5)]
        with self.assertRaises(ValueError):
            self.tta_manager.aggregate_results(
                predictions, 
                "invalid_method"
            )
