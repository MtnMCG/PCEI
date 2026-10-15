import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus

nexus = PCEINexus(vocab_size=50257, device="cpu", layers_text=2, layers_img=1, layers_audio=1)
nexus.neurogenesis.apply_structural_changes({"TEXT_L1": "SPAWN"})
keys = list(nexus.state_dict().keys())
text_layer_keys = [k for k in keys if "core.text_stack.layers" in k]
print("\n".join(text_layer_keys))
