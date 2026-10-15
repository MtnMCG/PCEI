import torch
from pathlib import Path

def peek_checkpoint(path):
    print(f"Peeking into: {path}")
    # Use weights_only=False to see the full structure
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        param_count = sum(p.numel() for p in state_dict.values())
        print(f"Total Parameters: {param_count:,}")
        
        # Check layers
        text_layers = set()
        for key in state_dict.keys():
            if 'text_stack.layers' in key:
                parts = key.split('.')
                text_layers.add(parts[parts.index('layers')+1])
        print(f"Text Layers: {len(text_layers)}")
        
        # Check d_model
        if 'core.text_stack.layers.0.mamba.in_proj.weight' in state_dict:
            d_model = state_dict['core.text_stack.layers.0.mamba.in_proj.weight'].shape[1]
            print(f"D_Model (Text): {d_model}")
            
    if 'optimizer_state_dict' in ckpt:
        print("Optimizer State: Present")
    else:
        print("Optimizer State: Missing")

if __name__ == "__main__":
    peek_checkpoint("checkpoints/pcei_500m.pt")
