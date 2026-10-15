import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
import random
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(str(Path("/Users/emerson/Desktop/PCEI /pcei_framework")))

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
    
    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        ids = [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]
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

def run_test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running final verification on {device}")
    
    tokenizer = PCEITokenizer(vocab_size=VOCAB_SIZE)
    nexus = PCEINexus(vocab_size=VOCAB_SIZE).to(device)
    
    # Load checkpoint
    ckpt_path = Path("/Users/emerson/Desktop/PCEI /pcei_framework/checkpoints/pcei_100m.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found!")
        return
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'])
    nexus.set_mode("ACTIVE")
    nexus.eval()
    
    # ═══════════════════════════════════════
    # TEST 1: IMAGE LABELING
    # ═══════════════════════════════════════
    print("\n" + "="*40)
    print(" TEST 1: IMAGE LABELING (Grounding)")
    print("="*40)
    
    db_path = Path("/Users/emerson/Desktop/PCEI /pcei_framework/data/processed/training_database.jsonl")
    dataset = PCEIDataset(db_path, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)
    
    image_records = [r for r in dataset.records if r.get('image_path')]
    if not image_records:
        print("No image records found!")
    else:
        root_path = Path("/Users/emerson/Desktop/PCEI /pcei_framework")
        image_ingestor = ImageIngestor()
        samples = random.sample(image_records, min(2, len(image_records)))
        
        for record in samples:
            img_path = root_path / record['image_path']
            print(f"\nID: {record['id']}")
            print(f"Original: {record['text']}")
            
            try:
                img_tensor = image_ingestor.load_image(img_path).to(device)
                text_input = tokenizer.encode("").unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, _, _, _ = nexus.core(
                        text_input=text_input,
                        img_input=img_tensor,
                        diffusion_timesteps=3,
                        slow_state=nexus.ambient_field.slow_state,
                        mean_confidence=0.95
                    )
                    
                    if logits is not None:
                        # greedy decoding
                        pred_ids = torch.argmax(logits[0], dim=-1)
                        prediction = tokenizer.decode(pred_ids).strip()
                        print(f"Beast:    {prediction}")
            except Exception as e:
                print(f"Error: {e}")

    # ═══════════════════════════════════════
    # TEST 2: CHAT TEST
    # ═══════════════════════════════════════
    print("\n" + "="*40)
    print(" TEST 2: CHAT (Hello)")
    print("="*40)
    
    inputs = ["hello", "describe your architecture"]
    for prompt in inputs:
        print(f"\nUser: {prompt}")
        text_input = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, t_latent, _, _ = nexus.core(
                text_input=text_input,
                img_input=None,
                audio_input=None,
                diffusion_timesteps=10,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=0.95
            )
            
            if logits is not None:
                pred_ids = torch.argmax(logits[0], dim=-1)
                response = tokenizer.decode(pred_ids).strip()
                # Clean up response (take first sentence or limited tokens)
                print(f"Beast: {response[:200]}...")
            else:
                print("Beast: [Quiet]")

if __name__ == "__main__":
    run_test()
