import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus

ckpt_path = Path("checkpoints/pcei_micro.pt")
t, i, a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
print(f"Topology: {t} Text | {i} Img | {a} Audio")

# Let's see what t_layers actually was
import torch
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt.get('model_state_dict', ckpt)
t_layers = set()
for key in state_dict.keys():
    if "core.text_stack.layers." in key:
        t_layers.add(key.split("core.text_stack.layers.")[1].split(".")[0])
print(f"t_layers set: {t_layers}")
