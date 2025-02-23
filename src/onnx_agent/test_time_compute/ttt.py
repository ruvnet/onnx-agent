import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List

class TTTOptimizer:
    """Test-Time Training Optimizer"""
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.initial_state = None
        
    def setup(self, model: nn.Module, learning_rate: float = 1e-4) -> None:
        """Setup optimizer with model"""
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=1, 
            threshold=1e-5, verbose=True
        )
        self.initial_state = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute adaptation loss on batch"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
            
        self.model.eval()  # Use eval mode for adaptation
        outputs = self.model(batch)
        
        # Modified loss to be more sensitive to weight changes
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(1).mean()
        l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.model.parameters())
        return entropy + l2_reg
        
    def adaptation_step(self, batch: torch.Tensor) -> float:
        """Perform single adaptation step"""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Call setup() first.")
            
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()
        
    def adapt(self, batch: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run full adaptation process"""
        history = {
            "losses": [],
            "converged": False,
            "iterations": 0,
            "initial_loss": None,
            "final_loss": None
        }
        
        # Initial loss
        initial_loss = self.compute_loss(batch).item()
        history["initial_loss"] = initial_loss
        history["losses"].append(initial_loss)
        
        # Adaptation loop
        for i in range(config["max_iterations"]):
            loss = self.adaptation_step(batch)
            history["losses"].append(loss)
            
            # Check convergence
            if i > 0:
                loss_diff = abs(history["losses"][-1] - history["losses"][-2])
                if loss_diff < config["convergence_threshold"]:
                    history["converged"] = True
                    break
                    
            # Update learning rate
            self.scheduler.step(loss)
            
        history["iterations"] = len(history["losses"])
        history["final_loss"] = history["losses"][-1]
        
        return history
        
    def reset(self) -> None:
        """Reset model to initial state"""
        if self.initial_state is None:
            raise ValueError("No initial state saved. Call setup() first.")
            
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.initial_state[name])
