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

    def forward(self, x: torch.Tensor, time: torch.Tensor, is_training: bool = False, cache=None, slow_state=None) -> torch.Tensor:
        """
        If we are in open training mode (which PCEI always is), standard gradient flow 
        handles adaptation. But during 'inference' or to accelerate adaptation to a specific 
        hard prompt, the TTT inner-loop runs first.
        """
        if is_training or self.num_steps == 0 or not torch.is_grad_enabled():
            return self.module(x, time, cache=cache, slow_state=slow_state)

        # MPS-Safe Test-Time Training (TTT) Inner Loop
        original_weights = {name: param.clone() for name, param in self.module.named_parameters() if param.requires_grad}
        
        for _ in range(self.num_steps):
            out = self.module(x, time, cache=cache, slow_state=slow_state)
            
            # Self-supervised proxy loss: Reconstruction (don't distort your own input)
            loss = torch.nn.functional.mse_loss(out, x.detach())
            
            # Safe gradient computation
            params = [p for n, p in self.module.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            
            # Apply safe SGD step (stripping NaNs for MPS stability)
            for param, grad in zip(params, grads):
                if grad is not None:
                    safe_grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    safe_grad = torch.clamp(safe_grad, min=-1.0, max=1.0)
                    param.data.copy_(param.data - self.lr * safe_grad)
                    
        # Final adapted forward pass for the actual response
        final_out = self.module(x, time, cache=cache, slow_state=slow_state)
        
        # Restore global weights to prevent permanent drift
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in original_weights:
                param.data.copy_(original_weights[name])
                
        return final_out

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
            
        # Soft clamp output to prevent pre-norm explosion (NaN during backward)
        out = torch.tanh(out / 10.0) * 10.0
            
        return self.norm(out)

