import torch
import torch.nn as nn
from typing import Dict, Optional
import math

from src.modules.ternary import TernaryLinear


class DreamSeeder(nn.Module):
    """
    Full-spec DreamSeeder.
    Generates directed noise seeds shaped by the AmbientField's slow_state.
    """
    def __init__(self, d_model_text: int = 256, d_model_img: int = 512, d_model_audio: int = 128):
        super().__init__()
        
        self.intent_dim = d_model_text
        
        # Dream policy networks: TernaryNet reading from field state
        self.dream_policy_text = nn.Sequential(
            TernaryLinear(self.intent_dim, self.intent_dim),
            nn.SiLU(),
            TernaryLinear(self.intent_dim, d_model_text),
        )
        self.dream_policy_img = nn.Sequential(
            TernaryLinear(self.intent_dim, self.intent_dim),
            nn.SiLU(),
            TernaryLinear(self.intent_dim, d_model_img),
        )
        self.dream_policy_audio = nn.Sequential(
            TernaryLinear(self.intent_dim, self.intent_dim),
            nn.SiLU(),
            TernaryLinear(self.intent_dim, d_model_audio),
        )
        
        # Learned dream_strength: controls how much the field influences the noise.
        self.dream_strength = nn.Parameter(torch.tensor(0.01))
        self.anneal_rate = 0.001
        self.active_intent: Optional[torch.Tensor] = None
        
    def prime(self, intent_vector: torch.Tensor):
        """Sets the active mental anchor for subsequent generations."""
        self.active_intent = intent_vector.detach()

    def clear(self):
        """Clears the active mental anchor."""
        self.active_intent = None
        
    def compute_field_summary(self, slow_state: torch.Tensor) -> torch.Tensor:
        """Takes the consolidated latent field and ensures it matches intent_dim."""
        # Shield: Ensure no NaNs enter the summary
        slow_state = torch.nan_to_num(slow_state, nan=0.0)
        
        if slow_state.size(-1) != self.intent_dim:
            if slow_state.size(-1) < self.intent_dim:
                return torch.nn.functional.pad(slow_state, (0, self.intent_dim - slow_state.size(-1)))
            return slow_state[..., :self.intent_dim]
        return slow_state

    def forward(self, 
                intent_vector: Optional[torch.Tensor] = None,
                batch_size: int = 1,
                seq_lens: Dict[str, int] = {},
                slow_state: Optional[torch.Tensor] = None,
                mean_confidence: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Generates physics-informed noise seeds shaped by the AmbientField.
        """
        # Use active intent if no specific intent vector is passed
        if intent_vector is None:
            intent_vector = self.active_intent
            
        if intent_vector is None:
            # Fallback to zero if absolutely no intent is found
            device = slow_state.device if slow_state is not None else torch.device("cpu")
            intent_vector = torch.zeros(batch_size, self.intent_dim, device=device)
            
        device = intent_vector.device
        seeds = {}
        
        # Anneal dream_strength (Section 9: anneals upward with confidence)
        if self.training:
            # Shield: Prevent NaN confidence from poisoning the dream_strength
            safe_conf = mean_confidence if not math.isnan(mean_confidence) else 0.01
            target_strength = max(0.01, float(safe_conf))
            with torch.no_grad():
                self.dream_strength.data += self.anneal_rate * (target_strength - self.dream_strength.data)
                self.dream_strength.data.clamp_(min=0.01, max=1.0) # FLOOR at 0.01
        
        # Get field-based modulation
        if slow_state is not None:
            field_summary = self.compute_field_summary(slow_state).to(device)
            if field_summary.size(0) != batch_size:
                field_summary = field_summary.expand(batch_size, -1)
        else:
            field_summary = torch.zeros(batch_size, self.intent_dim, device=device)
        
        # Combined input: intent (prompt) + field (memory)
        combined = intent_vector + field_summary * self.dream_strength
        
        # Dream policy generates modality-specific noise modulators (Mean and Std)
        if 'text' in seq_lens:
            # Field shapes the distribution
            field_mod = self.dream_policy_text(combined).unsqueeze(1)  # (B, 1, d_text)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0 # Bound to prevent explosion
            # Use field_mod to shape the standard deviation of the noise
            std = torch.exp(field_mod * 0.1) * self.dream_strength
            noise = torch.randn(batch_size, seq_lens['text'], field_mod.size(2), device=device)
            seeds['text'] = noise * std + field_mod * 0.01
            
        if 'img' in seq_lens:
            field_mod = self.dream_policy_img(combined).unsqueeze(1)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0
            std = torch.exp(field_mod * 0.1) * self.dream_strength
            noise = torch.randn(batch_size, seq_lens['img'], field_mod.size(2), device=device)
            seeds['img'] = noise * std + field_mod * 0.01
            
        if 'audio' in seq_lens:
            field_mod = self.dream_policy_audio(combined).unsqueeze(1)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0
            std = torch.exp(field_mod * 0.1) * self.dream_strength
            noise = torch.randn(batch_size, seq_lens['audio'], field_mod.size(2), device=device)
            seeds['audio'] = noise * std + field_mod * 0.01
            
        return seeds
