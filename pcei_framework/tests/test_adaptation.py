import torch
import pytest


from src.modules.adaptation import TTTAdapter, FastKANMixer
from src.modules.diffusion import DiffusionBlock

def test_fast_kan_mixer_shape_and_grad():
    channels = 64
    batch, seq = 4, 16
    
    mixer = FastKANMixer(channels=channels, degree=3)
    x = torch.randn(batch, seq, channels, requires_grad=True)
    
    out = mixer(x)
    assert out.shape == x.shape, f"FastKANMixer expected shape {x.shape}, got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input x did not receive gradient"
    assert mixer.coeffs.grad is not None, "Polynomial coefficients did not receive gradient"

def test_ttt_adapter_updates_weights():
    # Verify that TTT adapter adapts output while preserving base parameters
    d_model = 128
    batch, seq = 2, 8
    
    block = DiffusionBlock(d_model=d_model)
    initial_weight_sum = sum(p.sum().item() for p in block.parameters())
    
    adapter = TTTAdapter(module=block, lr=0.1, num_steps=3)
    
    x = torch.randn(batch, seq, d_model)
    time = torch.zeros(batch)
    
    with torch.enable_grad():
        out = adapter(x, time, is_training=False)
    
    final_weight_sum = sum(p.sum().item() for p in block.parameters())
    
    # Base parameters should be safely restored to prevent weight drift
    assert torch.isclose(torch.tensor(initial_weight_sum), torch.tensor(final_weight_sum), atol=1e-5), \
        "TTTAdapter should restore original parameters after forward pass"
    assert out.shape == (batch, seq, d_model), f"Expected shape {(batch, seq, d_model)}, got {out.shape}"

