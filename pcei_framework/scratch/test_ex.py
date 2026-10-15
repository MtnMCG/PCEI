import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus

ckpt_path = Path("checkpoints/pcei_micro.pt")
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Loaded fine!")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
