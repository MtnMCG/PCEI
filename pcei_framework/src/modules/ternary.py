import math
import torch
import torch.nn as nn
from torch.autograd import Function


class TernaryQuantize(Function):
    """
    BitNet 1.58b Metabolic Ternary {-1, 0, +1} with Threshold-Aware STE.

    Forward:
        Quantizes continuous weights into strictly 3 discrete states:
            W_q = +1.0  if  W >  t_low
            W_q =  0.0  if |W| <= t_low  (pruned / metabolic sparsity)
            W_q = -1.0  if  W < -t_low

    Backward:
        - weight: STE straight-through estimator (grad_weight = grad_output).
        - t_low:  Soft Gaussian boundary density gradient dynamically adjusting
                  pruning pressure based on weight concentration near threshold.
        - t_high (deprecated/legacy): Returns zero gradient if provided to maintain
                  full backward compatibility with legacy 5-state callers.
    """
    @staticmethod
    def forward(ctx, weight, t_low, t_high=None):
        abs_weight = weight.abs()
        # Strictly {-1.0, 0.0, +1.0}
        quantized = torch.sign(weight) * (abs_weight > t_low).to(weight.dtype)
        ctx.save_for_backward(weight, t_low)
        ctx.has_t_high = (t_high is not None)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        weight, t_low = ctx.saved_tensors
        abs_weight = weight.abs()

        # Weight gradient: standard STE
        grad_weight = grad_output

        # Threshold gradient: Soft Gaussian density at boundary zone
        bandwidth = t_low.detach().clamp(min=1e-4) * 0.3
        density_low = torch.exp(-0.5 * ((abs_weight - t_low) / bandwidth) ** 2)
        grad_t_low = -(grad_output * density_low).sum()

        if getattr(ctx, 'has_t_high', False):
            return grad_weight, grad_t_low, torch.zeros_like(t_low)
        return grad_weight, grad_t_low


class TernaryLinear(nn.Module):
    """
    BitNet 1.58b Linear Layer with Dynamic Threshold Scaling.
    
    Weights are stored in continuous full-precision / bfloat16 for gradient
    accumulation, and quantized dynamically to {-1.0, 0.0, +1.0} during forward passes.
    """
    def __init__(self, in_features, out_features, bias=True, initial_threshold=None, **kwargs):
        super(TernaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.get_default_dtype()))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Dynamic BitNet 1.58b threshold calibration
        # gamma is the mean absolute value of the weight matrix
        with torch.no_grad():
            gamma = self.weight.abs().mean().clamp(min=1e-6)
            if initial_threshold is not None:
                init_low = torch.tensor(initial_threshold, dtype=torch.get_default_dtype())
                init_high = torch.tensor(initial_threshold * 3.0, dtype=torch.get_default_dtype())
            elif kwargs.get('threshold_factor') is not None:
                tf = float(kwargs['threshold_factor'])
                init_low = tf * gamma
                init_high = (tf * 3.0) * gamma
            else:
                # BitNet 1.58b dynamic threshold scaling: threshold_low = 0.5 * gamma
                init_low = 0.5 * gamma
                init_high = 1.5 * gamma

        # Active BitNet decision boundary: weights below this are pruned (0.0)
        self.threshold_low = nn.Parameter(init_low.clone().detach().to(dtype=torch.get_default_dtype()))
        # Preserved parameter for seamless backward compatibility with existing checkpoint files
        self.threshold_high = nn.Parameter(init_high.clone().detach().to(dtype=torch.get_default_dtype()))
        self.raw_scale = nn.Parameter(torch.ones(1, dtype=torch.get_default_dtype()) / math.sqrt(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.get_default_dtype()))
        else:
            self.register_parameter('bias', None)

    @property
    def gamma(self) -> torch.Tensor:
        """Weight matrix mean absolute value (BitNet 1.58b scale factor)."""
        return self.weight.abs().mean()

    @property
    def quantized_weight(self) -> torch.Tensor:
        """Returns the strictly {-1.0, 0.0, +1.0} quantized weight matrix."""
        t_low = torch.abs(self.threshold_low) + 1e-8
        return TernaryQuantize.apply(self.weight, t_low)

    def calibrate_thresholds(self, scale_low: float = 0.5, scale_high: float = 1.5):
        """
        BitNet 1.58b dynamic threshold calibration:
        gamma = weight.abs().mean()
        threshold_low = scale_low * gamma
        threshold_high = scale_high * gamma
        """
        with torch.no_grad():
            g = self.gamma.clamp(min=1e-6)
            self.threshold_low.copy_(scale_low * g)
            self.threshold_high.copy_(scale_high * g)

    def forward(self, input):
        t_low = torch.abs(self.threshold_low) + 1e-8
        quantized_weight = TernaryQuantize.apply(self.weight, t_low)
        scale = torch.clamp(self.raw_scale.abs(), min=1e-5)
        return nn.functional.linear(input, quantized_weight * scale, self.bias)


