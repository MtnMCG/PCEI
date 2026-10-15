import torch
import torch.nn as nn
from typing import Tuple


from src.modules.diffusion import DiffusionBlock

class PlasticLayer(nn.Module):
    """
    Wraps the core computational block (DiffusionBlock) with physical properties:
    1. Grid Coordinates (x, y, z) for topological shifting and distance rendering
    2. slow_field_confidence: controls the layer's plasticity/effective learning rate
    """
    def __init__(self, layer_id: str, d_model: int, initial_pos: Tuple[float, float, float], base_lr: float = 1e-4, bidirectional: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = d_model
        
        # Core computation
        self.block = DiffusionBlock(d_model=d_model, bidirectional=bidirectional)
        
        # Physical coordinates (x, y, z).
        # We store these as tensors so they can be logged with the model, but they 
        # do not require gradients during standard backprop. The TopologyShifter moves them.
        self.register_buffer('pos_x', torch.tensor([float(initial_pos[0])]))
        self.register_buffer('pos_y', torch.tensor([float(initial_pos[1])]))
        self.register_buffer('pos_z', torch.tensor([float(initial_pos[2])]))
        
        # Subconscious linkage: represents how consolidated this layer's patterns are (0.0 to 1.0)
        self.register_buffer('slow_field_confidence', torch.tensor([0.0]))
        
        # The base learning rate for parameters in this specific layer
        self.base_lr = base_lr

    @property
    def effective_lr(self) -> float:
        """
        Calculates the effective learning rate modulated by the slow field.
        As confidence approaches 1.0, learning rate approaches 0.
        This provides physics-based prevention of catastrophic forgetting.
        """
        confidence = self.slow_field_confidence.item()
        return self.base_lr * (1.0 - confidence)

    def get_position(self) -> Tuple[float, float, float]:
        return (self.pos_x.item(), self.pos_y.item(), self.pos_z.item())

    def update_position(self, delta_x: float, delta_y: float):
        """Called by TopologyShifter to physically migrate the layer."""
        self.pos_x += delta_x
        self.pos_y += delta_y

    def adjust_learning_rates(self, optimizer: torch.optim.Optimizer):
        """
        Iterates over the optimizer's param groups, finding parameter groups where
        all parameters belong to this layer, and dynamically updates their learning rate 
        based on the layer's current slow_field_confidence.
        """
        conf = self.slow_field_confidence.item()
        plasticity_factor = max(0.0, 1.0 - conf)
        my_param_ids = {id(p) for p in self.parameters()}
        
        for group in optimizer.param_groups:
            group_param_ids = {id(p) for p in group['params']}
            if group_param_ids and group_param_ids.issubset(my_param_ids):
                base = group.get('initial_lr', self.base_lr)
                group['lr'] = base * plasticity_factor

    def forward(self, x: torch.Tensor, time: torch.Tensor, cache=None, slow_state=None) -> torch.Tensor:
        """
        Accepts the noisy latents and timesteps, passes them to the internal DiffusionBlock.
        """
        return self.block(x, time, cache=cache, slow_state=slow_state)

