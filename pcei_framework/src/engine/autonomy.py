import torch
import torch.nn as nn
from typing import Optional

class ResponseGate(nn.Module):
    """
    Autonomy component for the TextStack.
    Instead of outputting text synchronously at the last diffusion timestep,
    the PCEI model only outputs text when the latent representation has "settled"
    across timesteps (i.e., when energy/variance drops below a threshold),
    giving it dynamic "thinking time".
    """
    def __init__(self, d_model: int, settlement_threshold: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.settlement_threshold = settlement_threshold
        
        # Keep track of the previous timestep's latent to measure change (velocity)
        self.prev_latent = None

    def check_settlement(self, current_latent: torch.Tensor) -> bool:
        """
        Calculates the difference between the current diffusion step's latent
        and the previous one. If the representation has stopped changing rapidly,
        the gate opens.
        """
        if self.prev_latent is None:
            self.prev_latent = current_latent.detach().clone()
            return False
            
        # Calculate Mean Absolute Error (velocity of the latent)
        delta = torch.abs(current_latent - self.prev_latent).mean().item()
        
        # Update memory for next tick
        self.prev_latent = current_latent.detach().clone()
        
        # If the latent is stable, we have formed a solid thought
        if delta < self.settlement_threshold:
            return True
            
        return False
        
    def reset(self):
        self.prev_latent = None


class OutputRouter(nn.Module):
    """
    Controls whether the model is in 'Active' or 'Witness' mode.
    
    In Witness mode, the model ingests text, images, and audio, and its layers
    migrate and update slow_field_confidence, but the TextStack is muted 
    (no output is generated). This allows continuous background learning.
    """
    def __init__(self):
        super().__init__()
        self.mode = "ACTIVE" # or "WITNESS"

    def set_mode(self, mode: str):
        assert mode in ["ACTIVE", "WITNESS"]
        self.mode = mode

    def route_output(self, text_logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Decides whether to yield the logits to the user or swallow them.
        """
        if self.mode == "WITNESS" or text_logits is None:
            return None
        return text_logits
