import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from PIL import Image
import numpy as np
import textwrap

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.pcei import PCEINexus
from src.core.ingestors import PCEIDataset, ImageIngestor
import tiktoken

# Constants
MAX_SEQ_LEN = 512
VOCAB_SIZE = 50257

class PCEITokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256
    
    def encode(self, text: str, pad: bool = False, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        ids = [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]
        if pad:
            ids = ids[:max_len]
            while len(ids) < max_len:
                ids.append(self.pad_id)
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        raw_ids = [tok_id.item() for tok_id in ids if tok_id.item() != self.pad_id]
        raw_ids = [0 if tok_id == (self.vocab_size - 1) else tok_id for tok_id in raw_ids]
        try:
            return self.enc.decode(raw_ids)
        except Exception:
            return "[Decode Error]"

def run_visual_inference():
    device = "cpu"
    print(f"Running visual inference on {device}")
    
    tokenizer = PCEITokenizer(vocab_size=VOCAB_SIZE)
    nexus = PCEINexus(vocab_size=VOCAB_SIZE).to(device)
    
    # Load checkpoint
    ckpt_path = ROOT / "checkpoints" / "pcei_micro.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found!")
        return
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'])
    nexus.set_mode("ACTIVE")  # CRITICAL: Allow text stack to output decision
    nexus.eval()
    
    # Load dataset
    db_path = ROOT / "data" / "processed" / "training_database.jsonl"
    dataset = PCEIDataset(db_path, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)
    
    # Filter for image records
    image_records = [r for r in dataset.records if r.get('image_path')]
    if not image_records:
        print("No image records found in dataset!")
        return
        
    # Pick 4 random samples
    samples = random.sample(image_records, min(4, len(image_records)))
    
    fig, axes = plt.subplots(len(samples), 2, figsize=(15, 6 * len(samples)))
    if len(samples) == 1: axes = [axes]
    
    root_path = ROOT
    image_ingestor = ImageIngestor()


    for i, record in enumerate(samples):
        img_path = root_path / record['image_path']
        original_label = record['text']
        
        print(f"Processing {record['id']}...")
        
        from src.modules.plasticity import PlasticLayer
        conf_vals = []
        for _, module in nexus.named_modules():
            if isinstance(module, PlasticLayer):
                conf_vals.append(module.slow_field_confidence.item())
        actual_mean_conf = sum(conf_vals) / max(len(conf_vals), 1)
        print(f"Model Physics State: Confidence={actual_mean_conf:.4f}")
        # Load and process image for model
        try:
            img_tensor = image_ingestor.load_image(img_path).to(device) # (1, num_patches, patch_dim)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # We need a BLANK prompt for multimodal labeling, matching Phase 2 training
        # If we force an ID prefix, we are in a distribution the model hasn't seen with images.
        prompt_text = "" 
        text_input = tokenizer.encode(prompt_text).unsqueeze(0).to(device)

        # Model Inference (Autoregressive loop instead of parallel sampling)
        current_text = text_input
        generated_ids = []
        with torch.no_grad():
            for _ in range(50): # Max 50 tokens for captioning
                logits, _, _, _ = nexus.core(
                    text_input=current_text,
                    img_input=img_tensor,
                    audio_input=None,
                    diffusion_timesteps=3, 
                    slow_state=nexus.ambient_field.slow_state,
                    mean_confidence=actual_mean_conf
                )
                
                if logits is None: break
                
                next_token_logits = logits[0, -1, :] / 0.7
                # simple repetition penalty
                for past_id in set(generated_ids):
                    next_token_logits[past_id] /= 1.5
                    
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == tokenizer.eos_id: break
                generated_ids.append(next_token.item())
                
                current_text = torch.cat([current_text, next_token.unsqueeze(0)], dim=1)
                
            predicted_text = tokenizer.decode(torch.tensor(generated_ids)).strip()
            
            print(f"  PREDICTION: {predicted_text}")
            
        # Plotting
        ax_img = axes[i][0]
        ax_text = axes[i][1]
        
        # Original Image
        try:
            img = Image.open(img_path).convert('RGB')
            ax_img.imshow(img)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center')
            
        ax_img.set_title(f"Image: {record['id']}")
        ax_img.axis('off')
        
        # Text Comparison
        wrapped_original = textwrap.fill(original_label, width=50)
        wrapped_predicted = textwrap.fill(predicted_text[:300], width=50)
        
        comparison_text = f"ORIGINAL LABEL:\n{wrapped_original}\n\n"
        comparison_text += f"BEAST DEPICTION:\n{wrapped_predicted}"
        
        ax_text.text(0.05, 0.8, comparison_text, fontsize=12, verticalalignment='top', family='monospace')
        ax_text.axis('off')

    plt.suptitle("PCEI 'Unleashed Beast v2' - Multimodal Labeling Test", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = ROOT / "src" / "logs" / "visual_inference_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved inference comparison to {output_path}")

if __name__ == "__main__":
    run_visual_inference()
