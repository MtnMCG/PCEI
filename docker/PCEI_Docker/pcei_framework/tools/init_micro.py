import torch
import argparse
from pathlib import Path
from src.pcei import PCEINexus

def initialize_micro():
    print("═" * 60)
    print("  PCEI Micro-Nexus Initialization")
    print("  Goal: Real-time emergence testing")
    print("═" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Scale down to ~15M params
    nexus = PCEINexus(
        vocab_size=50257,
        device=device,
        d_model_text=128,  # Scaled down from 512
        d_model_img=128,   # Scaled down from 512
        d_model_audio=64,  # Scaled down from 256
        layers_text=2,     # Scaled down from 6
        layers_img=1,
        layers_audio=1
    )
    
    total_params = sum(p.numel() for p in nexus.parameters())
    print(f"[Model] Micro-Nexus Initialized: {total_params:,} parameters")
    
    ckpt_path = Path("checkpoints/pcei_micro.pt")
    ckpt_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': nexus.state_dict(),
        'global_step': 0,
        'config': {
            'd_model_text': 128,
            'd_model_img': 128,
            'd_model_audio': 64,
            'layers_text': 2,
            'layers_img': 1,
            'layers_audio': 1
        }
    }, ckpt_path)
    
    print(f"[Saved] Random-Ternary Foundation at {ckpt_path}")

if __name__ == "__main__":
    initialize_micro()
