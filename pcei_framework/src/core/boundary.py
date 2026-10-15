import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BoundaryLayer(nn.Module):
    """
    Full-sequence bidirectional projection layer between two modality stacks.
    
    Preserves temporal structure via F.interpolate for dynamic sequence length
    adaptation. This means any sequence length can be exchanged — no fixed-size
    linear bottleneck.
    
    E.g. TextStack (seq_len=16, d=256) ↔ ImageStack (seq_len=4, d=512)
    """
    def __init__(self, d_model_upper: int, d_model_lower: int,
                 seq_len_upper: int = 16, seq_len_lower: int = 16):
        super().__init__()
        self.d_model_upper = d_model_upper
        self.d_model_lower = d_model_lower
        self.seq_len_upper = seq_len_upper
        self.seq_len_lower = seq_len_lower
        
        # Channel projections
        self.proj_down = nn.Linear(d_model_upper, d_model_lower, bias=False)
        self.proj_up = nn.Linear(d_model_lower, d_model_upper, bias=False)
        
        # Learnable scale factors for temporal mixing (applied after interpolation)
        self.temporal_scale_down = nn.Parameter(torch.ones(1))
        self.temporal_scale_up = nn.Parameter(torch.ones(1))
        
        # Buffers for thread-safe exchange
        self.buffer_down = None
        self.buffer_up = None

    def _adapt_temporal(self, x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        """
        Adapts sequence length using F.interpolate (linear mode).
        This is dynamic and handles any input/output sequence length.
        
        x: (batch, seq_len_source, d_model)
        Returns: (batch, target_seq_len, d_model)
        """
        if x.size(1) == target_seq_len:
            return x
        
        # F.interpolate expects (B, C, L) format
        x_t = x.transpose(1, 2)  # (B, d_model, seq_len_source)
        x_t = F.interpolate(x_t, size=target_seq_len, mode='linear', align_corners=False)
        return x_t.transpose(1, 2)  # (B, target_seq_len, d_model)

    def send_down(self, x_upper: torch.Tensor, target_seq_len: int = None):
        """
        Upper stack pushes its FULL activation sequence down.
        Projects channels d_upper → d_lower and adapts seq_len dynamically.
        """
        if target_seq_len is None:
            target_seq_len = self.seq_len_lower
            
        projected = self.proj_down(x_upper)
        adapted = self._adapt_temporal(projected, target_seq_len)
        self.buffer_down = adapted * self.temporal_scale_down

    def receive_down(self) -> Optional[torch.Tensor]:
        data = self.buffer_down
        self.buffer_down = None
        return data

    def send_up(self, x_lower: torch.Tensor, target_seq_len: int = None):
        """Lower stack pushes its FULL activation sequence up."""
        if target_seq_len is None:
            target_seq_len = self.seq_len_upper
            
        projected = self.proj_up(x_lower)
        adapted = self._adapt_temporal(projected, target_seq_len)
        self.buffer_up = adapted * self.temporal_scale_up

    def receive_up(self) -> Optional[torch.Tensor]:
        data = self.buffer_up
        self.buffer_up = None
        return data
            
    def clear_buffers(self):
        self.buffer_down = None
        self.buffer_up = None
