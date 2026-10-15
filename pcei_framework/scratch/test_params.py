import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus

nexus = PCEINexus(
    vocab_size=50257,
    device="cpu",
    d_model_text=384,
    d_model_img=384,
    d_model_audio=192,
    layers_text=6,
    layers_img=3,
    layers_audio=3
)
print(f"Total Params: {sum(p.numel() for p in nexus.parameters()) / 1e6:.2f} M")
