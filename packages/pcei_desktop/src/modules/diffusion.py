import torch
import torch.nn as nn
import math
from typing import Optional


from src.modules.mamba import MambaBlock, MambaConfig
from src.modules.ternary import TernaryLinear
from src.modules.adaptation import FastKANMixer

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionBlock(nn.Module):
    """
    A unified block that takes a noisy continuous latent, integrates timestep
    information, and processes dependencies linearly across the sequence via Mamba.
    Uses TernaryNet as the primary projector layer.
    """
    def __init__(self, d_model: int, d_state: int = 16, bidirectional: bool = True):
        super().__init__()
        self.d_model = d_model
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Mamba replaces typical Multi-Head Attention to ensure linear time, 
        # continuous space, and local interaction.
        mamba_config = MambaConfig(d_model=d_model, d_state=d_state, bidirectional=bidirectional)
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
        self.norm1_post = nn.LayerNorm(d_model)
        self.norm2_post = nn.LayerNorm(d_model)

        # Ambient Field Global Broadcast (Zero-Gated Residual Adapter)
        self.ambient_gate = nn.Parameter(torch.zeros(1))
        self.ambient_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.ambient_proj.weight)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cache=None, slow_state: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, d_model) - the noisy latents
        time: (batch,) - the diffusion timesteps
        slow_state: Optional (batch, d_model) or (d_model,) - the subconscious AmbientField state
        """
        # 1. Compute timestep embedding and broadcast to sequence (only when time > 0)
        # When time == 0 (clean state / autoregressive text), skip injecting t_emb
        # to prevent adding a sequence-invariant DC bias vector that collapses token variance.
        is_clean_state = False
        if time is None:
            is_clean_state = True
        elif isinstance(time, (int, float)) and time == 0:
            is_clean_state = True
        elif torch.is_tensor(time) and (time == 0).all():
            is_clean_state = True

        if not is_clean_state:
            t_emb = self.time_mlp(time) # (batch, d_model)
            t_emb = t_emb.unsqueeze(1).expand_as(x) # (batch, seq_len, d_model)
            x = x + t_emb
        
        # Dual-pathway: Mamba sequence modeling + Ternary MLP reasoning
        # Pre-LN architecture with branch normalization & residual scaling:
        # Normalizes the branch outputs (preventing Mamba/MLP energy explosion)
        # while keeping the residual stream unnormalized to preserve token diversity & identity flow.
        residual_scale = 0.2
        
        # 1. Mamba sequence modeling branch
        m_out = self.mamba(self.norm1(x), cache=cache)
        m_out = self.norm1_post(m_out)
        x = x + m_out * residual_scale
        
        # 2. Ternary MLP reasoning branch
        mlp_out = self.ternary_mlp(self.norm2(x))
        mlp_out = self.norm2_post(torch.tanh(mlp_out))
        x = x + mlp_out * residual_scale

        # 3. Ambient Field Global Broadcast (Global Workspace Connection)
        if slow_state is not None:
            s = slow_state
            while s.dim() > 1 and s.size(0) != x.size(0):
                s = s.mean(dim=0)
            if s.dim() == 1:
                s_vec = s.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            elif s.dim() == 2:
                s_vec = s.unsqueeze(1)               # (batch, 1, d_model)
            else:
                s_vec = s
            
            s_vec = s_vec.to(dtype=x.dtype, device=x.device)
            if s_vec.size(0) != x.size(0):
                s_vec = s_vec.expand(x.size(0), -1, -1)
                
            ambient_broadcast = self.ambient_proj(s_vec)
            # Zero-initialized tanh gate guarantees exact 0.0 impact until trained
            x = x + torch.tanh(self.ambient_gate) * ambient_broadcast
        
        return x

