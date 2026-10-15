import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from scripts.train import PCEITokenizer

def audit():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = PCEITokenizer()
    nexus = PCEINexus(device=device)
    
    # Load and Sanitize
    ckpt_path = Path("checkpoints/pcei_micro.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # Deep Scrub parameters on the device
    for p in nexus.parameters():
        p.data = torch.nan_to_num(p.data, nan=0.0)
    for b in nexus.buffers():
        b.data = torch.nan_to_num(b.data, nan=0.0)
        
    print("Model loaded and scrubbed.")
    
    tokens = tokenizer.encode("Numerical Audit Probe")
    
    try:
        nexus.train()
        with torch.enable_grad():
            # 1. Integrate input to get initial prompt latent
            prompt_latent = nexus.core.text_stack.integrate_input(tokens.unsqueeze(0).to(device))
            
            # 2. Add noise for response
            resp_noise = torch.randn(1, 32, 128, device=device)
            current_latent = torch.cat([prompt_latent, resp_noise], dim=1)
            
            print(f"Initial Latent Shape: {current_latent.shape}")
            
            # 3. Mirror the Denoising Loop
            for t in reversed(range(10)):
                print(f"  Step T={t}...")
                _, denoised, _, _, _, _ = nexus.core(
                    text_latent=current_latent,
                    slow_state=nexus.ambient_field.slow_state,
                    diffusion_timesteps=t
                )
                
                if torch.isnan(denoised).any():
                    print(f"    [!!!] NaN detected at T={t}!")
                    break
                
                current_latent = denoised
                
            print("Audit complete.")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    audit()
