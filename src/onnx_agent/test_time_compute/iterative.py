import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass

@dataclass
class IterationResult:
    """Result from iterative inference"""
    final_prediction: np.ndarray
    confidence: float
    num_iterations: int
    converged: bool
    prediction_history: List[np.ndarray]
    confidence_history: List[float]

class IterativeInference:
    """Iterative Inference Manager"""
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        
    def setup(self, model: nn.Module) -> None:
        """Setup with model"""
        self.model = model
        
    def compute_confidence(self, prediction: np.ndarray) -> float:
        """Compute confidence score for prediction"""
        # For classification, use max probability
        probs = torch.softmax(torch.from_numpy(prediction), dim=1)
        confidence = float(torch.max(probs, dim=1)[0].mean())
        return confidence
        
    def check_stability(self, 
                       predictions: List[np.ndarray],
                       window_size: int) -> bool:
        """Check if predictions are stable over window"""
        if len(predictions) < window_size:
            return False
            
        # Get predictions over window
        window = predictions[-window_size:]
        
        # Check if all predictions in window are the same
        reference = window[0].argmax(axis=1)
        return all(np.array_equal(p.argmax(axis=1), reference) 
                  for p in window[1:])
                  
    def run_inference(self,
                     x: np.ndarray,
                     config: Dict[str, Any]) -> IterationResult:
        """Run iterative inference"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
            
        prediction_history = []
        confidence_history = []
        
        # Initial prediction
        self.model.eval()
        with torch.no_grad():
            x_tensor = x if torch.is_tensor(x) else torch.from_numpy(x)
            prediction = self.model(x_tensor).cpu().numpy()
            
        prediction_history.append(prediction)
        confidence = self.compute_confidence(prediction)
        confidence_history.append(confidence)
        
        converged = False
        
        # Iteration loop
        for i in range(config["max_iterations"]):
            # Run inference
            with torch.no_grad():
                prediction = self.model(x_tensor).cpu().numpy()
                
            prediction_history.append(prediction)
            confidence = self.compute_confidence(prediction)
            confidence_history.append(confidence)
            
            # Check confidence threshold
            if confidence >= config["confidence_threshold"]:
                converged = True
                break
                
            # Check stability
            if self.check_stability(prediction_history, 
                                  config["stability_window"]):
                converged = True
                break
                
        # Subtract 1 from len since initial prediction doesn't count as an iteration
        num_iterations = len(prediction_history) - 1
        
        return IterationResult(
            final_prediction=prediction_history[-1],
            confidence=confidence_history[-1],
            num_iterations=num_iterations,
            converged=converged,
            prediction_history=prediction_history,
            confidence_history=confidence_history
        )
