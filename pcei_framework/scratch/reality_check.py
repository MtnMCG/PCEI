import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from scripts.train import PCEITokenizer

def stress_test():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = PCEITokenizer()
    nexus = PCEINexus(
        device=device,
        d_model_text=512,
        d_model_img=512,
        d_model_audio=256,
        layers_text=6,
        layers_img=3,
        layers_audio=2
    )
    
    # Load latest checkpoint
    checkpoint_path = Path("checkpoints/pcei_checkpoint.pt")
    if not checkpoint_path.exists():
        print("Error: Checkpoint not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    nexus.load_state_dict(checkpoint['model_state_dict'])
    nexus.eval()
    
    # NEW TEXT (Never seen by model)
    test_text = "Physical systems that exhibit emergent intelligence must adhere to local constraints, otherwise the binding force of consciousness dissolves into a flat parameter space."
    
    # Encode
    tokens = tokenizer.encode(test_text).to(device).unsqueeze(0) # (1, 512)
    
    # Create a mask for specific words: "intelligence", "consciousness", "constraints"
    # Let's find their indices
    raw_ids = tokenizer.enc.encode(test_text)
    # Mapping for PCEITokenizer quirks (0 -> high vocab)
    raw_ids = [tid if tid != 0 else (tokenizer.vocab_size - 1) for tid in raw_ids]
    
    # Let's mask tokens at indices 5 (intelligence), 10 (constraints), 16 (consciousness)
    # (Note: indices might vary based on BPE, but we'll mask a span)
    masked_tokens = tokens.clone()
    masked_indices = [5, 6, 11, 12, 17, 18] # Covering the key words
    
    for idx in masked_indices:
        masked_tokens[0, idx] = tokenizer.pad_id # Use PAD as MASK
        
    print(f"--- REALITY CHECK ---")
    print(f"Original: {test_text}")
    print(f"Masked:   {tokenizer.decode(masked_tokens[0])}")
    print(f"---------------------")
    
    with torch.no_grad():
        logits, _, _, _, _, _ = nexus.core(
            text_input=masked_tokens,
            diffusion_timesteps=1,
            slow_state=nexus.ambient_field.slow_state,
            mean_confidence=0.9
        )
        
    # Get predictions
    preds = torch.argmax(logits, dim=-1) # (1, 512)
    
    # Extract predicted words
    predicted_text = tokenizer.decode(preds[0])
    print(f"Model Fill: {predicted_text}")
    
    # Compare key words
    success = 0
    for idx in masked_indices:
        if preds[0, idx] == tokens[0, idx]:
            success += 1
            
    print(f"---------------------")
    print(f"Semantic Match: {success}/{len(masked_indices)} tokens")
    
if __name__ == "__main__":
    stress_test()
