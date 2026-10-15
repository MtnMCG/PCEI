import torch
import pytest
from src.modules.ternary import TernaryQuantize, TernaryLinear

def test_ternary_quantize_forward_three_state():
    # Create weights with clear magnitude separation
    w = torch.tensor([-0.8, -0.1, 0.0, 0.1, 0.8, 0.5, -0.5, 0.02, -0.02, 0.3])
    
    t_low = torch.tensor(0.1)
    quantized = TernaryQuantize.apply(w, t_low)
    
    # All values must be strictly -1.0, 0.0, or 1.0 (True BitNet 1.58b)
    unique_vals = torch.unique(quantized)
    for val in unique_vals:
        assert val.item() in [-1.0, 0.0, 1.0], f"Found non-ternary value {val} in quantized weights"

    # Magnitude must only be 0.0 or 1.0
    assert not (quantized.abs() == 2.0).any(), "Found forbidden 2.0 state in weights!"

    count_0 = (quantized == 0).sum().item()
    count_neg1 = (quantized == -1).sum().item()
    count_pos1 = (quantized == 1).sum().item()
    assert count_0 > 0, "No pruned (0) weights found"
    assert count_neg1 > 0, "No inhibitory (-1) weights found"
    assert count_pos1 > 0, "No excitatory (+1) weights found"

def test_ternary_quantize_backward_ste():
    # Test Straight-Through Estimator (STE)
    w = torch.tensor([-0.8, -0.1, 0.0, 0.1, 0.8], requires_grad=True)
    t_low = torch.tensor(0.1)
    
    quantized = TernaryQuantize.apply(w, t_low)
    
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
    quantized = layer.quantized_weight
    
    unique_vals = torch.unique(quantized)
    allowed_vals = [-1.0, 0.0, 1.0]
    for val in unique_vals:
        assert val.item() in allowed_vals, f"Found non-ternary value {val}"
    assert not (quantized.abs() == 2.0).any(), "Found forbidden 2.0 state in TernaryLinear!"

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

