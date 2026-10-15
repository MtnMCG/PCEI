import torch
from pathlib import Path

def integrity_check():
    ckpt_path = Path("checkpoints/pcei_checkpoint.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    weights = checkpoint['model_state_dict']
    
    print(f"--- TERNARY INTEGRITY CHECK ---")
    
    # Check a weight layer (not a position layer)
    weight_keys = [k for k in weights.keys() if 'weight' in k]
    if not weight_keys:
        print("No weight layers found!")
        return
        
    target_key = weight_keys[0]
    w = weights[target_key]
    
    unique_vals = torch.unique(w)
    print(f"Layer: {target_key}")
    print(f"Unique values found: {len(unique_vals)}")
    print(f"Sample values: {unique_vals[:10].tolist()}")
    
    is_ternary = True
    for val in unique_vals:
        if not torch.isclose(val, torch.tensor(-1.0)) and \
           not torch.isclose(val, torch.tensor(0.0)) and \
           not torch.isclose(val, torch.tensor(1.0)):
            is_ternary = False
            break
            
    if is_ternary:
        print("✅ VERIFIED: Weights are TERNARY (-1, 0, 1) but stored as floats.")
    else:
        print("❌ CORRUPTED: Weights are DENSE FLOATS. The physics is broken.")
        # Calculate how "un-ternary" they are
        non_zero = w[w != 0]
        if len(non_zero) > 0:
            print(f"Mean of non-zero weights: {non_zero.abs().mean().item():.4f}")

if __name__ == "__main__":
    integrity_check()
