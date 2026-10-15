import torch
import pytest


from src.modules.mamba import MambaConfig, MambaBlock
from src.modules.diffusion import DiffusionBlock

def test_mamba_block_shape():
    batch, seq_len, d_model = 4, 32, 128
    
    config = MambaConfig(d_model=d_model)
    block = MambaBlock(config)
    
    x = torch.randn(batch, seq_len, d_model)
    out = block(x)
    
    assert out.shape == x.shape, f"Mamba expected out shape {x.shape}, got {out.shape}"

def test_diffusion_block_shape():
    batch, seq_len, d_model = 4, 32, 128
    
    block = DiffusionBlock(d_model=d_model)
    
    x = torch.randn(batch, seq_len, d_model)
    t = torch.randint(0, 1000, (batch,)) # random diffusion timesteps
    
    out = block(x, t)
    
    assert out.shape == x.shape, f"Diffusion expected out shape {x.shape}, got {out.shape}"

def test_diffusion_block_gradient_flow():
    batch, seq_len, d_model = 2, 16, 64
    
    block = DiffusionBlock(d_model=d_model)
    
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    t = torch.randint(0, 1000, (batch,))
    
    out = block(x, t)
    
    # Calculate a dummy loss
    loss = out.sum()
    loss.backward()
    
    # Check that gradients flow back to x without NaNs
    assert x.grad is not None, "Gradient to input x is None"
    assert not torch.isnan(x.grad).any(), "Gradient to input x contains NaNs"


def test_diffusion_block_autoregressive_zero_time():
    """Verify that at t=0 (autoregressive text mode), t_emb is bypassed and sequence diversity is preserved."""
    batch, seq_len, d_model = 2, 16, 64
    block = DiffusionBlock(d_model=d_model)
    
    # Input with distinct tokens
    x = torch.randn(batch, seq_len, d_model)
    t_zero = torch.zeros(batch, dtype=torch.long)
    
    out_zero = block(x, t_zero)
    assert out_zero.shape == x.shape
    
    # Verify that adjacent tokens in out_zero are not collapsed to 1.0 cosine similarity
    sim = torch.nn.functional.cosine_similarity(out_zero[0, :-1], out_zero[0, 1:], dim=-1).mean().item()
    assert sim < 0.95, f"Expected distinct tokens at t=0, but got collapsed cosine similarity {sim}"
    
    # When t > 0, t_emb should be computed and output should differ
    t_pos = torch.ones(batch, dtype=torch.long) * 5
    out_pos = block(x, t_pos)
    assert not torch.allclose(out_zero, out_pos), "Output at t=0 and t=5 should differ due to diffusion conditioning"

