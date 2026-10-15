import torch
import torch.nn as nn
import math


from src.modules.mamba import MambaBlock, MambaConfig
from src.modules.ternary import TernaryLinear
from src.modules.adaptation import FastKANMixer

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionBlock(nn.Module):
    """
    A unified block that takes a noisy continuous latent, integrates timestep
    information, and processes dependencies linearly across the sequence via Mamba.
    Uses TernaryNet as the primary projector layer.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Mamba replaces typical Multi-Head Attention to ensure linear time, 
        # continuous space, and local interaction.
        mamba_config = MambaConfig(d_model=d_model, d_state=d_state)
        self.mamba = MambaBlock(mamba_config)
        
        # Ternary FFN acting as the "expert" layer per block
        # Uses FastKANMixer (learnable polynomial activation) per spec Milestone 4
        self.kan_activation = FastKANMixer(channels=d_model * 4, degree=3)
        self.ternary_mlp = nn.Sequential(
            TernaryLinear(d_model, d_model * 4),
            self.kan_activation,
            TernaryLinear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        """
        x: (batch, seq_len, d_model) - the noisy latents
        time: (batch,) - the diffusion timesteps
        """
        # 1. Compute timestep embedding and broadcast to sequence
        t_emb = self.time_mlp(time) # (batch, d_model)
        t_emb = t_emb.unsqueeze(1).expand_as(x) # (batch, seq_len, d_model)
        
        # 2. Add time embedding to latent
        x = x + t_emb
        
        # Dual-pathway: Mamba sequence modeling + Ternary MLP reasoning
        # Each path is pre-normed and the result is post-normed to prevent energy explosion
        m_out = self.mamba(self.norm1(x))
        x = x + m_out
        x = self.norm1(x) # Post-Mamba Stabilizer
        
        # Ternary MLP reasoning block
        mlp_out = self.ternary_mlp(self.norm2(x))
        
        # Metabolic Constraint: We bound the MLP output to prevent runaway growth
        # Tanh is spec-compliant as it represents a biological saturation limit
        x = x + torch.tanh(mlp_out)
        x = self.norm2(x) # Post-MLP Stabilizer
        
        return x
