import math
import torch
import torch.nn as nn
from torch.autograd import Function


class TernaryQuantize(Function):
    """
    Spec-Correct Metabolic Ternary {0, 1, 2} with threshold-aware STE.

    Forward:  Hard ternary quantization {0, 1, 2}.
    Backward:
      - weight:      STE — raw gradient passes straight through (standard).
      - t_low:       Soft gradient based on density of weights near the low
                     boundary. If many weights are piled just above t_low,
                     t_low should rise to prune them. If the boundary zone
                     is empty, t_low gets no signal (correct — don't move
                     a boundary that isn't doing anything).
      - t_high:      Same logic for the high boundary. Weights near t_high
                     provide signal to adjust how aggressively weights
                     saturate to max.

    The boundary gradient is computed as a soft Gaussian density estimate
    centered on each threshold — a differentiable approximation of
    "how much weight mass is at this decision boundary."
    """
    @staticmethod
    def forward(ctx, weight, t_low, t_high):
        abs_weight = weight.abs()
        # Use torch.where instead of in-place boolean indexing.
        # In-place masked assignment (tensor[mask] = val) is a known MPS crash
        # when kernel shapes change between steps (index out of bounds on Metal).
        # torch.where is functionally identical and fully MPS-safe.
        quantized = torch.sign(weight) * torch.where(abs_weight <= t_low,
                                torch.zeros_like(weight),          # pruned
                    torch.where(abs_weight > t_high,
                                torch.full_like(weight, 2.0),      # strong
                                torch.ones_like(weight)))          # partial
        # Save for backward
        ctx.save_for_backward(weight, t_low, t_high)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        weight, t_low, t_high = ctx.saved_tensors
        abs_weight = weight.abs()

        # ── Weight gradient: standard STE ──────────────────────────────
        grad_weight = grad_output

        # ── Threshold gradients: boundary density signal ────────────────
        # Bandwidth for boundary sensitivity (fraction of threshold gap)
        bandwidth = (t_high - t_low).detach().clamp(min=1e-4) * 0.3

        # Gaussian density of weight mass near each threshold
        # High density near t_low → t_low is actively gating many weights
        density_low  = torch.exp(-0.5 * ((abs_weight - t_low)  / bandwidth) ** 2)
        density_high = torch.exp(-0.5 * ((abs_weight - t_high) / bandwidth) ** 2)

        # Gradient flows through the boundary weighted by output gradient magnitude
        # sign: if grad pushes output up, and we're at the low boundary,
        # reducing t_low would move those weights from 0→1 (adding signal).
        # The boundary should receive pressure proportional to that choice.
        grad_t_low  = -(grad_output * density_low).sum()
        grad_t_high =  (grad_output * density_high).sum()

        return grad_weight, grad_t_low, grad_t_high


class TernaryLinear(nn.Module):
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
                # BitNet 1.58b dynamic threshold scaling:
                # threshold_low = 0.5 * gamma, threshold_high = 1.5 * gamma
                # Prevents ternary dead zones across arbitrary matrix dimensions
                init_low = 0.5 * gamma
                init_high = 1.5 * gamma

        # Learnable decision boundaries calibrated to BitNet 1.58b scale
        # threshold_low:  weights below this → pruned (0)
        # threshold_high: weights above this → strong (2)
        # The gap between them is the "partial" zone (1)
        self.threshold_low  = nn.Parameter(init_low.clone().detach().to(dtype=torch.get_default_dtype()))
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
        # Enforce ordering: t_low < t_high, both positive
        t_low  = torch.abs(self.threshold_low) + 1e-8
        t_high = t_low + torch.abs(self.threshold_high - self.threshold_low) + 1e-6

        quantized_weight = TernaryQuantize.apply(self.weight, t_low, t_high)
        scale = torch.clamp(self.raw_scale.abs(), min=1e-5)
        return nn.functional.linear(input, quantized_weight * scale, self.bias)

