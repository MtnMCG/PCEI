import torch

path = "checkpoints/pcei_500m.pt"
checkpoint = torch.load(path, map_location="cpu")

print("Original Thresholds:")
for name, p in checkpoint['model_state_dict'].items():
    if "threshold_low" in name:
        print(f"  {name}: {p.item()}")
        # Fix the thresholds in place
        checkpoint['model_state_dict'][name] = torch.tensor(0.005)
    elif "threshold_high" in name:
        checkpoint['model_state_dict'][name] = torch.tensor(0.015)

print("\nSaving patched checkpoint...")
torch.save(checkpoint, path)
print("Done.")
