import torch
import pytest
from src.modules.ternary import TernaryQuantize, TernaryLinear

def test_ternary_quantize_forward_012():
    # Create weights with clear magnitude separation
    w = torch.tensor([-0.8, -0.1, 0.0, 0.1, 0.8, 0.5, -0.5, 0.02, -0.02, 0.3])
    
    t_low = torch.tensor(0.1)
    t_high = torch.tensor(0.5)
    quantized = TernaryQuantize.apply(w, t_low, t_high)
    
    # All magnitude values must be exactly 0.0, 1.0, or 2.0
    unique_vals = torch.unique(quantized.abs())
    for val in unique_vals:
        assert val.item() in [0.0, 1.0, 2.0], f"Found non-metabolic value {val} in quantized weights"

    count_0 = (quantized.abs() == 0).sum().item()
    count_1 = (quantized.abs() == 1).sum().item()
    count_2 = (quantized.abs() == 2).sum().item()
    assert count_0 > 0, "No pruned (0) weights found"
    assert count_1 > 0, "No partial (1) weights found"
    assert count_2 > 0, "No strong (2) weights found"

def test_ternary_quantize_backward_ste():
    # Test Straight-Through Estimator (STE)
    w = torch.tensor([-0.8, -0.1, 0.0, 0.1, 0.8], requires_grad=True)
    t_low = torch.tensor(0.1)
    t_high = torch.tensor(0.5)
    
    quantized = TernaryQuantize.apply(w, t_low, t_high)
    
    loss = quantized.sum()
    loss.backward()
    
    # STE: gradient should pass through as 1.0 everywhere
    expected_grad = torch.ones_like(w)
    assert torch.allclose(w.grad, expected_grad), f"Expected {expected_grad}, got {w.grad}"

def test_ternary_linear_shape():
    layer = TernaryLinear(in_features=10, out_features=5)
    x = torch.randn(32, 10)  # batch of 32
    out = layer(x)
    assert out.shape == (32, 5), f"Expected shape (32, 5), got {out.shape}"

def test_ternary_linear_weights_are_metabolic():
    layer = TernaryLinear(in_features=10, out_features=5)
    x = torch.randn(32, 10)
    out = layer(x)
    
    # Manually quantize to check
    t_low = torch.abs(layer.threshold_low) + 1e-8
    t_high = t_low + torch.abs(layer.threshold_high - layer.threshold_low) + 1e-6
    quantized = TernaryQuantize.apply(layer.weight, t_low, t_high)
    
    unique_vals = torch.unique(quantized.abs())
    allowed_vals = [0.0, 1.0, 2.0]
    for val in unique_vals:
        assert val.item() in allowed_vals, f"Found non-metabolic value {val}"

def test_ternary_linear_inhibition_via_sign():
    """Verify that negative weight values preserve negative signs for inhibition."""
    layer = TernaryLinear(in_features=4, out_features=2, bias=False)
    
    with torch.no_grad():
        layer.weight[0, :] = -0.8  # Force channel 0 weights negative
        layer.weight[1, :] = 0.8   # Force channel 1 weights positive
    
    x = torch.ones(1, 4) * 0.5
    out = layer(x)
    
    # Channel 0 should be negative (inhibitory), Channel 1 positive (excitatory)
    assert out[0, 0].item() < 0, f"Inhibitory channel should be negative, got {out[0, 0].item()}"
    assert out[0, 1].item() > 0, f"Excitatory channel should be positive, got {out[0, 1].item()}"

