import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus
from scripts.train import PCEITokenizer

device = "cpu"
tokenizer = PCEITokenizer()
nexus = PCEINexus(vocab_size=50257, device=device, d_model_text=128, d_model_img=128, d_model_audio=64, layers_text=2, layers_img=1, layers_audio=1)

checkpoint_path = Path("checkpoints/pcei_micro.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)

for p in nexus.parameters():
    mask = torch.isnan(p) | torch.isinf(p)
    if mask.any(): p.data[mask] = 0.0

for name, p in nexus.named_parameters():
    if "weight" in name and p.numel() > 10:
        zero_ratio = (p.data == 0).float().mean().item()
        if zero_ratio > 0.90:
            import math
            import torch.nn as nn
            if "dt_proj" in name:
                nn.init.uniform_(p.data, -0.01, 0.01)
            elif "A_log" in name:
                A = torch.arange(1, p.size(1) + 1).float()
                p.data.copy_(torch.log(A.repeat(p.size(0), 1)))
            elif "D" in name:
                nn.init.ones_(p.data)
            elif p.dim() >= 2:
                nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
            else:
                nn.init.normal_(p.data, std=0.02)
    if "bias" in name and "dt_proj" in name:
        dt_init = torch.exp(torch.rand_like(p.data) * -4.0 - 2.0)
        inv_softplus = lambda x: torch.log(torch.exp(x) - 1)
        p.data.copy_(inv_softplus(dt_init))

for name, p in nexus.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        print(f"NaN/Inf in weight {name} BEFORE FORWARD PASS!")

