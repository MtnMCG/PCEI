import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus

def audit():
    device = "cpu"
    print("Initializing Nexus...")
    nexus = PCEINexus(
        vocab_size=50257, 
        device=device,
        d_model_text=512,
        d_model_img=512,
        d_model_audio=256,
        layers_text=6,
        layers_img=3,
        layers_audio=2
    )
    
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / "pcei_checkpoint.pt"
    if not ckpt_path.exists():
        print("No checkpoint found.")
        return
        
    print(f"Loading checkpoint {ckpt_path.name}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("\n" + "="*50)
    print("PCEI TOPOLOGY AUDIT")
    print("="*50)
    
    from src.modules.plasticity import PlasticLayer
    
    layers_data = []
    for name, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            pos = module.get_position()
            conf = module.slow_field_confidence.item()
            layers_data.append((module.layer_id, pos, conf))
            
    print(f"\nFound {len(layers_data)} PlasticLayers.\n")
    
    for layer_id, pos, conf in layers_data:
        print(f"Layer: {layer_id:<15} | Pos: (X: {pos[0]:.4f}, Y: {pos[1]:.4f}, Z: {pos[2]:.1f}) | Confidence: {conf:.4f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    audit()
