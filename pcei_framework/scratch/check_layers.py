import torch
from pathlib import Path

ckpt_path = Path("checkpoints/pcei_micro.pt")
if not ckpt_path.exists():
    print("Checkpoint not found!")
    exit(1)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt['model_state_dict']

# Count layers in each stack based on state_dict keys
text_layers = set()
img_layers = set()
audio_layers = set()

for key in state_dict.keys():
    if key.startswith("core.text_stack.layers."):
        layer_idx = key.split(".")[3]
        text_layers.add(layer_idx)
    elif key.startswith("core.img_stack.layers."):
        layer_idx = key.split(".")[3]
        img_layers.add(layer_idx)
    elif key.startswith("core.audio_stack.layers."):
        layer_idx = key.split(".")[3]
        audio_layers.add(layer_idx)

print(f"Text Stack Layers: {len(text_layers)}")
print(f"Image Stack Layers: {len(img_layers)}")
print(f"Audio Stack Layers: {len(audio_layers)}")

# Check saturation (confidence values) to see if we're close to neurogenesis
from collections import defaultdict
confidence_vals = []
for key, value in state_dict.items():
    if "slow_field_confidence" in key:
        confidence_vals.append(value.item())

if confidence_vals:
    mean_conf = sum(confidence_vals) / len(confidence_vals)
    print(f"Mean Confidence (Saturation): {mean_conf:.4f}")
    if mean_conf > 0.85:
        print("Neurogenesis threshold (0.85) exceeded! Model is ripe for layer growth.")
    else:
        print("Model is still absorbing current topology. Not quite ready for neurogenesis.")
