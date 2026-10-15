"""
Multimodal Generation Testing Tool
Loads a trained PCEI checkpoint and generates sample Text, Audio, and Image outputs to evaluate structural convergence.
"""
import torch
import argparse
from pathlib import Path
from src.pcei import PCEINexus

def test_generation(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model (Micro-Nexus scale)
    nexus = PCEINexus(
        vocab_size=50257,
        d_model_text=128, d_model_img=128, d_model_audio=64,
        layers_text=2, layers_img=1, layers_audio=1
    ).to(device)
    
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        nexus.load_state_dict(model_state, strict=False)
        print("Model weights loaded successfully.")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Running with untrained weights for API validation.")
        
    nexus.eval()
    
    print("\n--- Testing Text Generation ---")
    dummy_text = torch.randint(1, 50000, (1, 16)).to(device)
    
    # Needs valid context dimensions for embedding integration
    dummy_img = torch.randn(1, 3, 32, 32).to(device)
    dummy_aud = torch.randn(1, 80, 100).to(device)
    
    with torch.no_grad():
        text_logits, t_lat, i_lat, a_lat, img_recon, aud_recon = nexus.core(
            text_input=dummy_text,
            img_input=dummy_img,
            audio_input=dummy_aud
        )
        print(f"Text Output Shape: {text_logits.shape} -> Expected (1, N, 50257)")
        
    print("\n--- Testing Image Generation ---")
    if img_recon is not None:
        print(f"Image Output Latent Shape: {img_recon.shape}")
        print("Image decoder successfully produced visual patches.")
    else:
        print("Image generation returned None. Ensure image_input was provided.")
        
    print("\n--- Testing Audio Generation ---")
    if aud_recon is not None:
        print(f"Audio Output Latent Shape: {aud_recon.shape}")
        print("Audio decoder successfully produced mel frames.")
    else:
        print("Audio generation returned None. Ensure audio_input was provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/pcei_micro.pt")
    args = parser.parse_args()
    test_generation(args.ckpt)
