import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading on {device}...")

# Initialize standard Nexus
nexus = PCEINexus(
    vocab_size=50257, 
    device=device,
    d_model_text=128, layers_text=2,
    d_model_img=128, layers_img=1,
    d_model_audio=64, layers_audio=1
).to(device)

ckpt_path = Path("checkpoints/pcei_micro.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)

print(f"Current Text Layers: {len(nexus.core.text_stack.layers)}")

# Force Neurogenesis on the deepest text layer (TEXT_L1)
print("\nForcing structural expansion (Neurogenesis)...")
nexus.neurogenesis.apply_structural_changes({"TEXT_L1": "SPAWN"})

print(f"New Text Layers: {len(nexus.core.text_stack.layers)}")

# Save back to checkpoint
print("\nSaving expanded topology...")
# Need to clear optimizer state dict because the parameter count changed
torch.save({
    'model_state_dict': nexus.state_dict(),
    'optimizer_state_dict': {} # Reset optimizer since params changed
}, ckpt_path)

print("Neurogenesis complete. The ASSISTANT has grown.")
