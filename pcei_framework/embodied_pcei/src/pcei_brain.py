"""
embodied_pcei/src/pcei_brain.py

PCEI Embodied Brain — Actor-Critic Neural Policy for Physical Humanoid Control.
Integrates:
- Mamba SSM (O(1) state-space sequence modeling for temporal gait coordination)
- BitNet 1.58b Ternary MLPs with FastKAN non-linear motor reasoning
- AmbientField homeostatic modulation (stamina, health, hunger)
- Continuous Gaussian Actor & Value-Function Critic heads for PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from typing import Tuple, Optional, Dict, List

# Import PCEI architectural primitives
from src.modules.diffusion import DiffusionBlock
from src.modules.mamba import MambaConfig, MambaBlock
from src.modules.ternary import TernaryLinear
from src.engine.ambient import AmbientField


class PCEIActorCritic(nn.Module):
    """
    Embodied PCEI Policy & Value Network.
    
    Takes continuous observation vector (proprioception + raycasts + metabolism),
    projects it into the PCEI latent space, routes it through DiffusionBlocks
    (Mamba SSM + Ternary MLPs), modulates with AmbientField, and outputs
    continuous joint action distributions + value function estimates.
    """
    def __init__(self, obs_dim: int = 575, action_dim: int = 21,
                 d_model: int = 128, num_layers: int = 4, d_state: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        
        # 1. Sensory Ingestion (Continuous Observation → PCEI Latent Space)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Ambient Field for homeostatic metabolic modulation
        self.ambient_field = AmbientField(model=self, fast_decay=0.85, slow_decay=0.98)
        self.metabolic_proj = nn.Linear(3, d_model)
        
        # 3. PCEI Neural Backbone (Stack of DiffusionBlocks with Mamba + Ternary FastKAN)
        self.layers = nn.ModuleList([
            DiffusionBlock(d_model=d_model, d_state=d_state, bidirectional=False)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)
        
        # 4. Actor Head (Gaussian Policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, action_dim)
        )
        # Learnable log standard deviation for continuous Gaussian exploration
        # Clamped between -3.0 (std ~0.05) and -0.35 (std ~0.70) to prevent bang-bang saturation
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -0.6)
        
        # 5. Critic Head (State Value Function V(s))
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()

    def sanitize_log_std(self):
        """Sanitizes actor_log_std parameter tensor into safe continuous range [0.05, 0.70]."""
        with torch.no_grad():
            self.actor_log_std.clamp_(-3.0, -0.35)

    def _init_weights(self):
        """Orthogonal weight initialization for stable PPO policy learning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor output layer initialized with smaller gain for initial near-zero actions
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_mean[-1].bias)
        # Critic output layer
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def extract_features(self, obs: torch.Tensor, cache=None) -> torch.Tensor:
        """
        Routes observations through the PCEI Embodied Backbone.
        obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
        Returns: latent features (batch, d_model)
        """
        is_seq = (obs.dim() == 3)
        if not is_seq:
            obs = obs.unsqueeze(1) # (batch, 1, obs_dim)
            
        batch_size, seq_len, _ = obs.shape
        
        # 1. Encode observations to latents
        x = self.obs_encoder(obs) # (batch, seq_len, d_model)
        
        # 2. Extract metabolic telemetry (last 3 entries of observation)
        metabolic_signals = obs[:, :, -3:] # (batch, seq_len, 3)
        metabolic_latent = self.metabolic_proj(metabolic_signals.mean(dim=1)) # (batch, d_model)
        
        # Modulate AmbientField with internal metabolic state
        slow_state = self.ambient_field.slow_state
        if slow_state is not None:
            if slow_state.device != obs.device:
                slow_state = slow_state.to(obs.device)
            slow_state = slow_state + metabolic_latent
        else:
            slow_state = metabolic_latent
            
        # 3. Route through PCEI DiffusionBlocks (Mamba SSM + Ternary MLPs)
        time_zero = torch.zeros(batch_size, device=obs.device, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x, time=time_zero, cache=cache, slow_state=slow_state)
            
        x = self.norm_final(x)
        
        # Return last timestep representation
        if is_seq:
            return x
        return x.squeeze(1) # (batch, d_model)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False, cache=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution.
        Returns: (action, log_prob, value, entropy)
        """
        features = self.extract_features(obs, cache=cache)
        
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(torch.clamp(self.actor_log_std, -3.0, -0.35))
        
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
            
        # Sum log-probabilities across the 21 independent joint action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, value, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions during PPO gradient updates.
        Returns: (log_prob, entropy, value)
        """
        features = self.extract_features(obs)
        
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(torch.clamp(self.actor_log_std, -3.0, -0.35))
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        
        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the critic value function estimate V(s)."""
        features = self.extract_features(obs)
        return self.critic(features).squeeze(-1)

    def tick_ambient(self, latent: torch.Tensor, loss: float = 0.0):
        """Ticks the subconscious AmbientField with latent representation and scalar loss."""
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
        self.ambient_field.tick(latent, loss)

