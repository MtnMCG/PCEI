import torch
import torch.nn as nn
from pathlib import Path
import sys
import random
from PIL import Image
import numpy as np
import textwrap
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.core.ingestors import PCEIDataset, ImageIngestor, AudioIngestor
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

def generate_response(nexus, tokenizer, text_input=None, img_input=None, audio_input=None, max_new_tokens=25, temperature=0.8, top_p=0.9, diffusion_steps=10):
    device = next(nexus.parameters()).device
    
    if text_input is None or text_input.numel() == 0:
        text_input = tokenizer.encode(" ").unsqueeze(0).to(device)
    
    # Get current model confidence
    from src.modules.plasticity import PlasticLayer
    conf_vals = []
    for _, module in nexus.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    actual_mean_conf = sum(conf_vals) / max(len(conf_vals), 1)

    generated_ids = []
    current_text = text_input
    
    nexus.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _, _, _, _ = nexus.core(
                text_input=current_text,
                img_input=img_input,
                audio_input=audio_input,
                diffusion_timesteps=diffusion_steps, 
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=actual_mean_conf
            )
            
            if logits is None: break
            
            # Sampling: Top-P (Nucleus) Sampling
            next_token_logits = logits[0, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            
            # Apply repetition penalty to prevent "reconstruct reconstruct" loops
            if len(generated_ids) > 0:
                for past_id in set(generated_ids):
                    next_token_logits[past_id] /= 1.5
                    
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token.item())
            if next_token.item() == tokenizer.eos_id: break
            
            # Update current_text for next step (simple concat for prototype)
            # In a real autoregressive setup we'd use a kv-cache, here we just append
            if current_text.size(1) < MAX_SEQ_LEN:
                current_text = torch.cat([current_text, next_token.unsqueeze(0)], dim=1)
            else:
                current_text = torch.cat([current_text[:, 1:], next_token.unsqueeze(0)], dim=1)
                
    return tokenizer.decode(torch.tensor(generated_ids))

def run_suite(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"PCEI Comprehensive Test Suite | Device: {device}")
    
    tokenizer = PCEITokenizer(vocab_size=VOCAB_SIZE)
    
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(ckpt_path, args.layers_text, args.layers_img, args.layers_audio, args.d_model_text, args.d_model_img, args.d_model_audio)
    print(f"[Topology] Detected: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")

    # Initialize architecture
    nexus = PCEINexus(
        vocab_size=VOCAB_SIZE, 
        device=device,
        d_model_text=d_text, layers_text=layers_text,
        d_model_img=d_img, layers_img=layers_img,
        d_model_audio=d_audio, layers_audio=layers_audio
    ).to(device)
    
    # Load latest checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found!")
        return
        
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
    nexus.set_mode("ACTIVE")
    
    image_ingestor = ImageIngestor()
    audio_ingestor = AudioIngestor(max_duration=60.0) # Extended to capture more context

    db_path = Path(__file__).parent.parent / "data/processed/training_database.jsonl"
    dataset = None
    if db_path.exists():
        dataset = PCEIDataset(db_path, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)

    # --- MODE 1: VISUAL LABELING ---
    if args.mode in ("all", "visual"):
        print("\n--- TEST: Visual Labeling ---")
        if dataset is not None:
            image_records = [r for r in dataset.records if r.get('image_path')]
        
        if image_records:
            samples = random.sample(image_records, min(3, len(image_records)))
            for r in samples:
                img_path = Path(__file__).parent.parent / r['image_path']
                print(f"\nImage: {r['image_path']}")
                print(f"Label: {r['text']}")
                img_tensor = image_ingestor.load_image(img_path).to(device)
                prompt_tensor = tokenizer.encode("Caption:").unsqueeze(0).to(device)
                pred = generate_response(nexus, tokenizer, text_input=prompt_tensor, img_input=img_tensor, diffusion_steps=args.steps)
                print(f"ASSISTANT: {pred}")

    # --- MODE 2: AUDIO SIGNAL TESTING ---
    if args.mode in ("all", "audio"):
        print("\n--- TEST: Audio Signal Evaluation ---")
        if dataset is not None:
            audio_records = [r for r in dataset.records if r.get('audio_path')]
            if audio_records:
                samples = random.sample(audio_records, min(3, len(audio_records)))
                for r in samples:
                    audio_path = Path(__file__).parent.parent / r['audio_path']
                    print(f"\nAudio: {r['audio_path']}")
                    print(f"Label: {r['text']}")
                    if audio_path.exists():
                        audio_tensor = audio_ingestor.load_audio(audio_path).to(device)
                        # Ensure shape is (1, T, 80)
                        if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(0)
                        
                        # Prompt the model about the audio
                        prompt = tokenizer.encode("The sound feels like ").unsqueeze(0).to(device)
                        resp = generate_response(nexus, tokenizer, text_input=prompt, audio_input=audio_tensor, diffusion_steps=args.steps)
                        print(f"ASSISTANT RESPONSE: {resp}")
                    else:
                        print("Audio file not found.")

    # --- MODE 3: CHAT ---
    if args.mode in ("all", "chat"):
        print("\n--- TEST: Interactive Dialogue ---")
        print("Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\nYOU: ")
            except EOFError: break
            if user_input.lower() in ("exit", "quit"): break
            
            prompt = tokenizer.encode(user_input).unsqueeze(0).to(device)
            resp = generate_response(nexus, tokenizer, text_input=prompt, diffusion_steps=args.steps)
            print(f"ASSISTANT: {resp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "visual", "audio", "chat"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--d-model-text", type=int, default=1536)
    parser.add_argument("--d-model-img", type=int, default=512)
    parser.add_argument("--d-model-audio", type=int, default=256)
    parser.add_argument("--layers-text", type=int, default=24)
    parser.add_argument("--layers-img", type=int, default=4)
    parser.add_argument("--layers-audio", type=int, default=4)
    args = parser.parse_args()
    
    # Update generate_response call inside run_suite to use args.steps
    # (Need to modify the calls in run_suite)
    run_suite(args)
