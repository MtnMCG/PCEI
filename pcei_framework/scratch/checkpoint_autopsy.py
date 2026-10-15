import torch
from pathlib import Path

def autopsy():
    ckpt_path = Path("checkpoints/pcei_checkpoint.pt")
    if not ckpt_path.exists():
        print("No checkpoint found.")
        return
        
    print(f"--- CHECKPOINT AUTOPSY ---")
    print(f"File: {ckpt_path.name}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    print(f"Keys: {list(checkpoint.keys())}")
    if 'global_step' in checkpoint:
        print(f"Global Step: {checkpoint['global_step']}")
    
    if 'model_state_dict' in checkpoint:
        weights = checkpoint['model_state_dict']
        print(f"Layers found: {len(weights)}")
        
        # Check a few layers for non-zero values
        first_layer_key = list(weights.keys())[0]
        first_layer_weights = weights[first_layer_key]
        print(f"Sample Layer ({first_layer_key}) Mean: {first_layer_weights.mean().item():.6f}")
        print(f"Sample Layer ({first_layer_key}) Std:  {first_layer_weights.std().item():.6f}")
        
        if first_layer_weights.std() < 1e-6:
            print("WARNING: Weights appear to be UNINITIALIZED or ZEROED.")
        else:
            print("Weights appear to contain REAL DATA.")

if __name__ == "__main__":
    autopsy()
