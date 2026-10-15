import torch
import argparse
from pathlib import Path
from src.pcei import PCEINexus
import sys; sys.path.append(str(Path(__file__).parent.parent))
from src.config import PCEIConfig

def initialize_micro():
    print("═" * 60)
    print("  PCEI Micro-Nexus Initialization")
    print("  Goal: Real-time emergence testing")
    print("═" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    config = PCEIConfig(
        d_model_text=128, layers_text=2,
        d_model_img=128, layers_img=1,
        d_model_audio=64, layers_audio=1
    )
    nexus = PCEINexus(config=config, device=device)
    
    total_params = sum(p.numel() for p in nexus.parameters())
    print(f"[Model] Micro-Nexus Initialized: {total_params:,} parameters")
    
    ckpt_path = Path("checkpoints/pcei_micro.pt")
    ckpt_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': nexus.state_dict(),
        'global_step': 0,
        'config': vars(config) if hasattr(config, '__dict__') else config
    }, ckpt_path)
    
    print(f"[Saved] Random-Ternary Foundation at {ckpt_path}")

if __name__ == "__main__":
    initialize_micro()
