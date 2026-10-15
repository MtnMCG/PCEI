
import torch
import sys
import os

def check_checkpoint(path):
    print(f"--- Analyzing: {path} ---")
    if not os.path.exists(path):
        print("  File not found.")
        return
    
    try:
        data = torch.load(path, map_location='cpu')
        model_state = data.get('model_state_dict', {})
        
        nan_layers = []
        inf_layers = []
        zero_layers = []
        
        for name, param in model_state.items():
            if torch.isnan(param).any():
                nan_layers.append(name)
            if torch.isinf(param).any():
                inf_layers.append(name)
            if (param == 0).all():
                zero_layers.append(name)
                
        if nan_layers:
            print(f"  [CRITICAL] {len(nan_layers)} layers contain NaNs:")
            for l in nan_layers[:5]: print(f"    - {l}")
        else:
            print("  [OK] No NaNs detected.")
            
        if inf_layers:
            print(f"  [CRITICAL] {len(inf_layers)} layers contain Infs:")
            for l in inf_layers[:5]: print(f"    - {l}")
        else:
            print("  [OK] No Infs detected.")
            
        print(f"  Total params checked: {len(model_state)}")
        
    except Exception as e:
        print(f"  Failed to load: {e}")

if __name__ == "__main__":
    check_checkpoint("checkpoints/pcei_checkpoint_integration_complete.pt")
    check_checkpoint("checkpoints/pcei_checkpoint_bootstrap_complete.pt")
    check_checkpoint("checkpoints/pcei_checkpoint_latest.pt")
