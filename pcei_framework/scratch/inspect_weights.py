import torch

path = "checkpoints/pcei_500m.pt"
checkpoint = torch.load(path, map_location="cpu")

print("Checking weight magnitudes:")
for name, p in checkpoint['model_state_dict'].items():
    if "ternary_mlp.0.weight" in name and "text_stack.layers.0" in name:
        print(f"{name}:")
        print(f"  Shape: {p.shape}")
        print(f"  Max: {p.max().item():.5f}, Min: {p.min().item():.5f}")
        print(f"  Mean: {p.mean().item():.5f}, Std: {p.std().item():.5f}")
        print(f"  Zeros: {(p == 0).float().mean().item() * 100:.2f}%")
        
        # Test threshold of 0.005
        t_low = 0.005
        t_high = 0.015
        mask_neg = p < -t_low
        mask_pos = p > t_low
        mask_strong_neg = p < -t_high
        mask_strong_pos = p > t_high
        
        print(f"  If threshold is {t_low}:")
        print(f"    -1 (weak): {mask_neg.sum().item()} | +1 (weak): {mask_pos.sum().item()}")
        print(f"    -2 (strong): {mask_strong_neg.sum().item()} | +2 (strong): {mask_strong_pos.sum().item()}")
        
        break
