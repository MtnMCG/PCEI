import torch
import os

ckpt_path = 'checkpoints/pcei_micro.pt'
if not os.path.exists(ckpt_path):
    print("Checkpoint not found.")
    exit(1)

print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu')
sd = ckpt['model_state_dict']

cleaned_total = 0
keys_cleaned = 0

for k, v in sd.items():
    if torch.is_tensor(v):
        mask = torch.isnan(v) | torch.isinf(v)
        if mask.any():
            count = mask.sum().item()
            v[mask] = 0.0
            cleaned_total += count
            keys_cleaned += 1
            # print(f"  Scrubbed {k}: {count} NaNs")

print(f"Deep Scrub Complete. Cleaned {cleaned_total} NaNs across {keys_cleaned} keys.")
torch.save(ckpt, ckpt_path)
print("Saved clean checkpoint.")
