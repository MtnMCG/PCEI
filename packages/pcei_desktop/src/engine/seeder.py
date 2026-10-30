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
        """Takes the consolidated latent field and ensures it matches intent_dim.
        
        Always collapses to shape (1, intent_dim) so that expand(B, seq_len, -1)
        works correctly regardless of what shape slow_state arrives as from the field.
        """
        # Shield: Ensure no NaNs enter the summary
        slow_state = torch.nan_to_num(slow_state, nan=0.0)
        
        # Collapse any leading batch/seq dimensions to a single representative vector
        # e.g. (B, T, D) -> (1, D), (T, D) -> (1, D), (D,) -> (1, D)
        s = slow_state
        while s.dim() > 1:
            s = s.mean(dim=0)  # mean-pool each leading dimension
        s = s.unsqueeze(0)  # (D,) -> (1, D)
        
        # Align to intent_dim
        if s.size(-1) != self.intent_dim:
            if s.size(-1) < self.intent_dim:
                s = torch.nn.functional.pad(s, (0, self.intent_dim - s.size(-1)))
            else:
                s = s[..., :self.intent_dim]
        return s  # (1, intent_dim)

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
            dtype = slow_state.dtype if slow_state is not None else torch.float32
            intent_vector = torch.zeros(batch_size, self.intent_dim, device=device, dtype=dtype)
            
        device = intent_vector.device
        sys_dtype = intent_vector.dtype
        seeds = {}
        
        # Ensure intent_vector is 3D: (B, seq_len, d_model)
        if intent_vector.dim() == 2:
            intent_vector = intent_vector.unsqueeze(1)
        seq_len = intent_vector.size(1)
        
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
            # Cast both device AND dtype — slow_state may be CPU float32 from old checkpoints
            field_summary = self.compute_field_summary(slow_state).to(device=device, dtype=sys_dtype)
            while field_summary.dim() < 3:
                field_summary = field_summary.unsqueeze(0)
            field_summary = field_summary.expand(batch_size, seq_len, -1)
        else:
            field_summary = torch.zeros(batch_size, seq_len, self.intent_dim, device=device, dtype=sys_dtype)
        
        # Ensure intent_vector matches intent_dim
        if intent_vector.size(-1) != self.intent_dim:
            if intent_vector.size(-1) < self.intent_dim:
                intent_vector = torch.nn.functional.pad(intent_vector, (0, self.intent_dim - intent_vector.size(-1)))
            else:
                intent_vector = intent_vector[..., :self.intent_dim]

        # Combined input: intent (prompt) + field (memory) - Shape: (B, seq_len, d_model)
        combined = intent_vector + field_summary * self.dream_strength
        
        # Dream policy generates modality-specific noise modulators (Mean and Std)
        if 'text' in seq_lens:
            field_mod = self.dream_policy_text(combined)  # (B, seq_len, d_text)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0 # Bound to prevent explosion
            std = (torch.nn.functional.softplus(field_mod * 0.1) + 0.1) * self.dream_strength
            
            target_len = seq_lens['text']
            if field_mod.size(1) != target_len:
                field_mod = torch.nn.functional.interpolate(field_mod.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                std = torch.nn.functional.interpolate(std.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                
            noise = torch.randn(batch_size, target_len, field_mod.size(2), device=device, dtype=sys_dtype)
            seeds['text'] = noise * std + field_mod * 0.01
            
        if 'img' in seq_lens:
            field_mod = self.dream_policy_img(combined)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0
            std = (torch.nn.functional.softplus(field_mod * 0.1) + 0.1) * self.dream_strength
            
            target_len = seq_lens['img']
            if field_mod.size(1) != target_len:
                field_mod = torch.nn.functional.interpolate(field_mod.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                std = torch.nn.functional.interpolate(std.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)

            noise = torch.randn(batch_size, target_len, field_mod.size(2), device=device, dtype=sys_dtype)
            seeds['img'] = noise * std + field_mod * 0.01
            
        if 'audio' in seq_lens:
            field_mod = self.dream_policy_audio(combined)
            field_mod = torch.tanh(field_mod / 50.0) * 50.0
            std = (torch.nn.functional.softplus(field_mod * 0.1) + 0.1) * self.dream_strength
            
            target_len = seq_lens['audio']
            if field_mod.size(1) != target_len:
                field_mod = torch.nn.functional.interpolate(field_mod.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                std = torch.nn.functional.interpolate(std.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)

            noise = torch.randn(batch_size, target_len, field_mod.size(2), device=device, dtype=sys_dtype)
            seeds['audio'] = noise * std + field_mod * 0.01
            
        return seeds
