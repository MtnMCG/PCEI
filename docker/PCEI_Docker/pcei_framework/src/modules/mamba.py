import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaConfig:
    def __init__(self, d_model=256, d_inner=512, d_state=16, dt_rank="auto", 
                 d_conv=4, expand=2, bias=False, conv_bias=True):
        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_model * expand if d_inner in (None, "auto") else d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank != "auto" else math.ceil(d_model / 16)
        self.d_conv = d_conv
        self.bias = bias
        self.conv_bias = conv_bias

class MambaBlock(nn.Module):
    """
    Pure PyTorch fallback implementation of a single Mamba block.
    This is drastically slower than the official CUDA `mamba-ssm` but 
    allows us to run the PCEI architecture on an Apple Silicon (M4 Air) device.
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
        
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # Selective parameters
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Spec-Correct Initialization: Ensure dt starts in a healthy regime
        dt_init_std = config.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize bias to log(dt_min) to log(dt_max) (approx 0.001 to 0.1)
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(config.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-6)
        inv_softplus = lambda x: torch.log(torch.expm1(x))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus(dt_init))
        
        # State matrices A and D
        A = torch.arange(1, config.d_state + 1).float()
        # Shape: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A.repeat(config.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Bidirectional: runs a forward scan and a backward scan, sums the results.
        """
        batch, seq_len, _ = x.shape
        
        x_proj = self.in_proj(x)
        # Split into x and res
        x_inner, res = x_proj.split(self.config.d_inner, dim=-1)
        
        # Conv
        # x shape: (batch, seq_len, d_inner) -> (batch, d_inner, seq_len)
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :seq_len] # truncate padding
        x_inner = x_inner.transpose(1, 2)
        
        x_inner = self.act(x_inner)
        
        # SSM parameters
        x_dbl = self.x_proj(x_inner) # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (batch, seq_len, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Bidirectional Selective Scan
        y_fwd = self._selective_scan_direction(x_inner, dt, A, B, C, self.D, reverse=False)
        y_bwd = self._selective_scan_direction(x_inner, dt, A, B, C, self.D, reverse=True)
        y = y_fwd + y_bwd
        
        # Gate
        y = y * self.act(res)
        
        out = self.out_proj(y)
        return out
        
    def _selective_scan_direction(self, x, dt, A, B, C, D, reverse=False):
        """
        Autograd-safe selective scan. 
        Note: Force-moved to CPU to bypass persistent MPS memory allocation bugs 
        (Placeholder storage not allocated) in complex recurrent loops.
        """
        orig_device = x.device
        x = x.to("cpu")
        dt = dt.to("cpu")
        A = A.to("cpu")
        B = B.to("cpu")
        C = C.to("cpu")
        
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        h = x.new_zeros((batch, d_inner, d_state))
        outputs = []
        
        scan_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        for l in scan_indices:
            dt_l = dt[:, l].view(batch, d_inner, 1)
            dA_l = torch.exp(-dt_l * torch.exp(A).view(1, d_inner, d_state)).clamp(max=1.0)
            
            curr_x = x[:, l].view(batch, d_inner, 1).clamp(-10.0, 10.0)
            dB_l = (curr_x * dt_l) * B[:, l].view(batch, 1, d_state) 
            
            h = (dA_l * h) + dB_l
            
            # Metabolic Recurrence Clamp: Instead of z-scoring which breaks 
            # state decay, we use a magnitude cap to prevent explosion.
            h = h.clamp(-100.0, 100.0)
            
            if l % 8 == 0:
                h = torch.nan_to_num(h, nan=0.0, posinf=100.0, neginf=-100.0)
            
            y_l = (h * C[:, l].view(batch, 1, d_state)).sum(dim=-1)
            outputs.append(y_l)
            
        if reverse:
            outputs = outputs[::-1]
            
        y = torch.stack(outputs, dim=1)
        # Move back to original device (e.g. MPS)
        return (y + x * D.to("cpu")).to(orig_device)

