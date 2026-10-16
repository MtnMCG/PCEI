import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaConfig:
    def __init__(self, d_model=256, d_inner=None, d_state=16, dt_rank="auto", 
                 d_conv=4, expand=2, bias=False, conv_bias=True, bidirectional=True):
        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_model * expand if d_inner in (None, "auto") else d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank != "auto" else math.ceil(d_model / 16)
        self.d_conv = d_conv
        self.bias = bias
        self.conv_bias = conv_bias
        self.bidirectional = bidirectional

class MambaBlock(nn.Module):
    """
    Pure PyTorch MPS-compatible Mamba block.
    Uses a vectorized parallel-prefix scan instead of a Python loop —
    processes entire sequences in a small number of tensor ops.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        
        self.act = nn.SiLU()
        
        # Selective parameters
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Spec-Correct Initialization
        dt_init_std = config.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(config.d_inner, dtype=torch.get_default_dtype()) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-6)
        inv_softplus = lambda x: torch.log(torch.expm1(x))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus(dt_init))
        
        A = torch.arange(1, config.d_state + 1, dtype=torch.get_default_dtype())
        self.A_log = nn.Parameter(torch.log(A.repeat(config.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(config.d_inner, dtype=torch.get_default_dtype()))
        
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x, cache=None):
        """
        x: (batch, seq_len, d_model)
        If cache is provided and seq_len=1, executes fast O(1) recurrent step.
        Otherwise, executes parallel hardware-aware scan (bidirectional if configured).
        """
        if cache is not None and x.size(1) == 1:
            return self._step(x, cache)
            
        batch, seq_len, _ = x.shape
        
        x_proj = self.in_proj(x)
        x_inner, res = x_proj.split(self.config.d_inner, dim=-1)
        
        # Conv (depthwise, causal-ish)
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.act(x_inner)
        
        # Conv state cache population for prefill
        if cache is not None:
            pad_len = max(0, self.config.d_conv - seq_len)
            x_inner_cache = x_inner.transpose(1, 2)
            if pad_len > 0:
                x_inner_cache = F.pad(x_inner_cache, (pad_len, 0))
            cache['conv_state'] = x_inner_cache[:, :, -self.config.d_conv:]
            
        # SSM parameters
        x_dbl = self.x_proj(x_inner)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # (batch, seq, d_inner)
        A = -torch.exp(self.A_log)               # (d_inner, d_state)
        
        # Parallel Vectorized Selective Scan
        y = self._selective_scan_vectorized(x_inner, dt, A, B, C, self.D, reverse=False, cache=cache)
        
        if self.config.bidirectional:
            y_bwd = self._selective_scan_vectorized(x_inner, dt, A, B, C, self.D, reverse=True)
            y = y + y_bwd
            
        y = y * self.act(res)
        return self.out_proj(y)
        
    def _step(self, x, cache):
        """O(1) Recurrent step for lightning-fast autoregressive generation."""
        batch = x.size(0)
        
        x_proj = self.in_proj(x)
        x_inner, res = x_proj.split(self.config.d_inner, dim=-1)
        
        # Conv state cache
        if 'conv_state' not in cache:
            cache['conv_state'] = torch.zeros(batch, self.config.d_inner, self.config.d_conv, device=x.device, dtype=x.dtype)
            
        conv_state = cache['conv_state']
        conv_state = torch.cat([conv_state[:, :, 1:], x_inner.transpose(1, 2)], dim=-1)
        cache['conv_state'] = conv_state
        
        # Apply conv weight directly without padding
        x_inner = F.conv1d(conv_state, self.conv1d.weight, self.conv1d.bias, groups=self.config.d_inner)
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.act(x_inner)
        
        # SSM parameters
        x_dbl = self.x_proj(x_inner)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # SSM Recurrence cache
        if 'h' not in cache:
            cache['h'] = torch.zeros(batch, self.config.d_inner, self.config.d_state, device=x.device, dtype=x.dtype)
        
        h = cache['h']
        
        dA = torch.exp(dt.squeeze(1).unsqueeze(-1) * A.unsqueeze(0)).clamp(max=1.0)
        dB = x_inner.squeeze(1).unsqueeze(-1).clamp(-10.0, 10.0) * dt.squeeze(1).unsqueeze(-1) * B.squeeze(1).unsqueeze(1)
        
        # Stateful Update
        h = dA * h + dB
        cache['h'] = h
        
        y = (h * C.squeeze(1).unsqueeze(1)).sum(-1)
        y = y + x_inner.squeeze(1) * self.D
        y = y.unsqueeze(1)
        
        y = y * self.act(res)
        return self.out_proj(y)

    def _selective_scan_vectorized(self, x, dt, A, B, C, D, reverse=False, cache=None):
        """
        Vectorized selective scan — no Python loop over sequence positions.

        The SSM recurrence is:
            h_t = dA_t * h_{t-1} + dB_t
            y_t = (h_t * C_t).sum(-1) + D * x_t

        where:
            dA_t = exp(-dt_t * A)      per-token decay gates
            dB_t = x_t * dt_t * B_t   per-token input projection

        Using the log-space prefix-product trick:
            h_t = cum_dA_t * sum_{s=0}^{t} (dB_s / cum_dA_s)

        This computes h for all t simultaneously in ~8 tensor ops instead of
        seq_len Python iterations.
        """
        if reverse:
            x  = x.flip(1)
            dt = dt.flip(1)
            B  = B.flip(1)
            C  = C.flip(1)

        # ── Step 1: per-token gates and inputs ──────────────────────────────
        # A:  (d_inner, d_state)  →  (1, 1, d_inner, d_state)
        # dt: (batch, seq, d_inner) → (batch, seq, d_inner, 1)
        A_4d  = A.unsqueeze(0).unsqueeze(0)                          # (1, 1, d_inner, d_state)
        dt_4d = dt.unsqueeze(-1)                                      # (batch, seq, d_inner, 1)

        # dA[t] = exp(dt[t] * A), shape: (batch, seq, d_inner, d_state)
        # Note: A is already negative, so dt * A is negative, causing decay.
        dA = torch.exp(dt_4d * A_4d).clamp(max=1.0)

        # dB[t] = x[t] * dt[t] * B[t]
        x_4d = x.unsqueeze(-1).clamp(-10.0, 10.0)                    # (batch, seq, d_inner, 1)
        B_4d = B.unsqueeze(2)                                         # (batch, seq, 1, d_state)
        dB   = x_4d * dt_4d * B_4d                                   # (batch, seq, d_inner, d_state)

        # ── Step 2: Stable Chunked Associative Scan ────────────────
        # A pure python loop is too slow, but pure log-cumsum collapses.
        # We chunk the sequence to prevent float32 underflow while maintaining speed.
        seq_len = x.size(1)
        chunk_size = 64
        
        h = torch.zeros_like(dB)
        curr_h = torch.zeros_like(dB[:, 0])
        
        for i in range(0, seq_len, chunk_size):
            c_dA = dA[:, i:i+chunk_size]
            c_dB = dB[:, i:i+chunk_size]
            
            # Local scan within chunk
            log_dA = torch.log(c_dA.clamp(min=1e-8))
            cum_log_dA = torch.cumsum(log_dA, dim=1)
            cum_dA = torch.exp(cum_log_dA)
            
            weighted_dB = c_dB / cum_dA.clamp(min=1e-8)
            cum_weighted_dB = torch.cumsum(weighted_dB, dim=1)
            local_h = cum_dA * cum_weighted_dB
            
            # Carry over from previous chunks
            carry_h = curr_h.unsqueeze(1) * cum_dA
            c_h = local_h + carry_h
            
            h[:, i:i+chunk_size] = c_h
            curr_h = c_h[:, -1]
            
        h = h.clamp(-100.0, 100.0)
        h = torch.nan_to_num(h, nan=0.0, posinf=100.0, neginf=-100.0)
        
        if cache is not None and not reverse:
            cache['h'] = curr_h

        # ── Step 3: output projection ────────────────────────────────────────
        # C: (batch, seq, d_state) → (batch, seq, 1, d_state)
        C_4d = C.unsqueeze(2)
        y    = (h * C_4d).sum(dim=-1)                                 # (batch, seq, d_inner)

        if reverse:
            y = y.flip(1)

        return y + x * D
