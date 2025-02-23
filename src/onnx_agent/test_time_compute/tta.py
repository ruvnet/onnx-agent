import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

class TTAManager:
    """Test-Time Augmentation Manager"""
    
    def __init__(self):
        self.augmentations = {
            "horizontal_flip": self._horizontal_flip,
            "vertical_flip": self._vertical_flip,
            "rotate90": self._rotate90
        }
        
    def _horizontal_flip(self, x: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return np.flip(x, axis=3)
        
    def _vertical_flip(self, x: np.ndarray) -> np.ndarray:
        """Flip image vertically"""
        return np.flip(x, axis=2)
        
    def _rotate90(self, x: np.ndarray) -> np.ndarray:
        """Rotate image 90 degrees clockwise"""
        return np.rot90(x, k=1, axes=(2, 3))
    
    def apply_augmentation(self, x: np.ndarray, aug_name: str) -> np.ndarray:
        """Apply a single augmentation"""
        if aug_name not in self.augmentations:
            raise ValueError(f"Unknown augmentation: {aug_name}")
        return self.augmentations[aug_name](x)
    
    def apply_all_augmentations(self, x: np.ndarray, aug_names: List[str]) -> List[np.ndarray]:
        """Apply multiple augmentations"""
        return [self.apply_augmentation(x, name) for name in aug_names]
    
    def aggregate_results(self, predictions: List[np.ndarray], method: str = "mean") -> np.ndarray:
        """Aggregate predictions from multiple augmentations"""
        if not predictions:
            raise ValueError("Empty predictions list")
            
        if method == "mean":
            return np.mean(predictions, axis=0)
        elif method == "voting":
            predictions_array = np.array(predictions)
            return np.argmax(np.bincount(predictions_array.argmax(axis=2).flatten()))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    def register_augmentation(self, name: str, func: callable) -> None:
        """Register a custom augmentation"""
        if name in self.augmentations:
            raise ValueError(f"Augmentation {name} already exists")
        self.augmentations[name] = func
