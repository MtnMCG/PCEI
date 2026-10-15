import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable

class TTTAdapter(nn.Module):
    """
    Test-Time Training Adapter.
    Wraps a PlasticLayer (or any module). During inference, it uses a self-supervised
    loss (like entropy minimization on the outputs) to take `num_steps` gradient steps 
    on the current batch, adapting the weights to the specific prompt before yielding the 
    final response.
    
    In PCEI, this allows TernaryNet's latent shadow weights to shift slightly before 
    quantization for each prompt.
    """
    def __init__(self, module: nn.Module, lr: float = 1e-4, num_steps: int = 1):
        super().__init__()
        self.module = module
        self.lr = lr
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor, time: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        """
        If we are in open training mode (which PCEI always is), standard gradient flow 
        handles adaptation. But during 'inference' or to accelerate adaptation to a specific 
        hard prompt, the TTT inner-loop runs first.
        """
        if is_training or self.num_steps == 0:
            return self.module(x, time)

        # In pure Test-Time Training, we temporarily adapt the module just for this x
        # We need to save the original state to restore it if we don't want permanent 
        # adaptation, but in PCEI the thesis states ALL usage is training, so we may 
        # actually let these test-time gradients persist!
        
        # We need an inner optimizer just for this module
        inner_opt = optim.SGD(self.module.parameters(), lr=self.lr)
        
        # Ensure gradients are enabled even in torch.no_grad() contexts 
        # (Though PCEI should ideally never be run in no_grad mode)
        with torch.enable_grad():
            for _ in range(self.num_steps):
                inner_opt.zero_grad()
                
                # Forward pass
                out = self.module(x, time)
                
                # Self-supervised TTT loss:
                # Without ground truth, we can minimize the variance/entropy of the latent 
                # space to force the layer to find a stable representation of the input.
                # A common TTT loss for continuous latents is distance to a rolling mean, 
                # or simply L2 regularization of the output changes.
                
                # For this prototype, we minimize the L2 norm of the activations to encourage
                # sparsity and efficient representations (metabolic constraint).
                loss = out.pow(2).mean()
                
                loss.backward()
                inner_opt.step()
                
                # We need to detach x.grad if it was computed so the outer loop doesn't double-count
                if x.grad is not None:
                    x.grad.zero_()

        # Final forward pass with adapted weights
        return self.module(x, time)

class FastKANMixer(nn.Module):
    """
    Because the `pykan` library is notoriously slow and designed for 2D inputs,
    we implement a fast, learnable polynomial/spline activation fallback here.
    This serves as the 'KANMixer' requirement allowing dynamic activation shapes.
    """
    def __init__(self, channels: int, degree: int = 3):
        super().__init__()
        self.channels = channels
        self.degree = degree
        # Learnable coefficients for the polynomial: a + bx + cx^2 + dx^3
        # Shape: (degree + 1, channels)
        self.coeffs = nn.Parameter(torch.randn(degree + 1, channels) * 0.01)
        
        # Initialize to approximate identity: f(x) ≈ x
        with torch.no_grad():
            self.coeffs[0, :] = 0.0  # a = 0
            self.coeffs[1, :] = 1.0  # b = 1.0 -> b*x = x
        
        # Stabilize output to prevent downstream explosion
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_len, channels)
        # Clamp inputs to prevent polynomial explosion (x^3 with large x = NaN)
        x_clamped = x.clamp(-4.0, 4.0)
        
        out = torch.zeros_like(x)
        
        # Calculate polynomial sum: out = a + b*x + c*x^2 + d*x^3 ...
        for d in range(self.degree + 1):
            term = (x_clamped ** d) * self.coeffs[d].view(1, 1, self.channels)
            out = out + term
            
        return self.norm(out)

