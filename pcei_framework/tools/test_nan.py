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

checkpoint_path = Path("checkpoints/pcei_100m.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Same sanitizer as resonance_chat
for p in nexus.parameters():
    mask = torch.isnan(p) | torch.isinf(p)
    if mask.any(): p.data[mask] = 0.0

for b in nexus.buffers():
    mask = torch.isnan(b) | torch.isinf(b)
    if mask.any(): b.data[mask] = 0.0

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

nexus.train()

# 1. WITNESS PHASE
prompt_ids = tokenizer.enc.encode("hi", allowed_special={"<|endoftext|>"})
prompt_ids = [tid if tid != 0 else (tokenizer.vocab_size - 1) for tid in prompt_ids]
input_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)

def check_nan(name, t):
    if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
        print(f"NaN found at {name}!")
        return True
    return False

clean_latent = nexus.core.text_stack.integrate_input(input_tokens)
check_nan("integrate_input", clean_latent)

split = max(1, clean_latent.size(1) // 2)
prefix = clean_latent[:, :split, :]
target_clean = clean_latent[:, split:, :]
noise = torch.randn_like(target_clean) * 0.2
target_noisy = target_clean + noise
training_input = torch.cat([prefix, target_noisy], dim=1)

print("training_input contains NaN?", torch.isnan(training_input).any().item())

_, denoised_latent, _, _, _, _ = nexus.core(
    text_latent=training_input,
    diffusion_timesteps=1,
    slow_state=nexus.ambient_field.slow_state,
    mean_confidence=0.5
)

