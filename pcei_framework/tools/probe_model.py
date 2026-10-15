import torch
import sys
from pathlib import Path

def probe_checkpoint():
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "pcei_100m.pt"
    if not ckpt_path.exists():
        print("Checkpoint not found.")
        return
        
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint.get("model_state_dict", {})
    
    total_params = 0
    nan_count = 0
    inf_count = 0
    dead_layers = 0
    zero_params = 0
    
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            numel = tensor.numel()
            total_params += numel
            
            nans = torch.isnan(tensor).sum().item()
            infs = torch.isinf(tensor).sum().item()
            zeros = (tensor == 0).sum().item()
            
            nan_count += nans
            inf_count += infs
            zero_params += zeros
            
            if numel > 10 and (zeros / numel) > 0.9:
                dead_layers += 1
                print(f"Dead layer: {name} (Zeros: {zeros}/{numel})")
                
    print(f"Total parameters: {total_params}")
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    print(f"Zero params: {zero_params} ({zero_params/max(1, total_params):.2%} of total)")
    print(f"Dead layers (>90% zeros): {dead_layers}")
    
    if nan_count > 0 or inf_count > 0:
        print("Model is UNHEALTHY (contains NaNs or Infs).")
    elif dead_layers > 0:
        print("Model is UNHEALTHY (contains dead layers).")
    else:
        print("Model appears HEALTHY.")

if __name__ == "__main__":
    probe_checkpoint()
