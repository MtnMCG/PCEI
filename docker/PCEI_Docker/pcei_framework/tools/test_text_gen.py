import torch
import argparse
import sys
from pathlib import Path
from rich.console import Console

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus

class PCEITokenizer:
    def __init__(self, vocab_size: int = 50257):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> torch.Tensor:
        ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        ids = [tid if tid != 0 else (self.vocab_size - 1) for tid in ids]
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        raw_ids = [0 if tid == (self.vocab_size - 1) else tid for tid in ids.tolist() if tid != self.pad_id]
        return self.enc.decode(raw_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_500m.pt")
    parser.add_argument("--prompt", type=str, default="The universe is")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.9)
    args = parser.parse_args()

    console = Console()
    console.print(f"\n[bold blue]PCEI Inference Engine v2.0[/bold blue]")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(checkpoint_path, 2, 1, 1)
    nexus = PCEINexus(vocab_size=50257, device=device, d_model_text=d_text, d_model_img=d_img, d_model_audio=d_audio,
                      layers_text=layers_text, layers_img=layers_img, layers_audio=layers_audio).to(device)
    
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
        console.print(f"[green]Loaded {args.checkpoint_name}[/green]")
    
    tokenizer = PCEITokenizer()
    tokens = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    
    console.print(f"[bold yellow]PROMPT:[/bold yellow] {args.prompt}")
    print("GENERATED: ", end="", flush=True)
    
    generated = tokens
    nexus.eval()
    
    with torch.no_grad():
        for i in range(args.steps):
            # PCEI Forward Pass
            conf_vals = [m.slow_field_confidence.item() for m in nexus.core.modules() if hasattr(m, 'slow_field_confidence')]
            mean_conf = sum(conf_vals) / max(len(conf_vals), 1) if conf_vals else 0.5
            
            # Run 5-step diffusion for coherence (matches overnight training depth)
            device_type = 'cuda' if 'cuda' in str(device) else ('mps' if 'mps' in str(device) else 'cpu')
            amp_dtype = torch.float16 if device_type == 'mps' else (torch.bfloat16 if device_type == 'cuda' else torch.float32)
            
            with torch.autocast(device_type='cuda' if device_type == 'cuda' else 'cpu', dtype=amp_dtype, enabled=(device_type == 'cuda')):
                _, denoised, _, _, _, _ = nexus.core(
                    text_input=generated,
                    diffusion_timesteps=5,
                    slow_state=nexus.ambient_field.slow_state,
                    mean_confidence=mean_conf
                )
            
            logits = nexus.core.text_stack.decode_output(denoised)
            
            # Stronger Repetition Penalty for emerging models
            probs = torch.softmax(logits[0, -1, :] / args.temp, dim=-1)
            
            # Penalize any token that has already appeared
            for prev_token in generated[0]:
                probs[prev_token] *= 0.5 # Aggressive penalty to break the "rative" loops
            
            # Normalize and sample
            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            if next_token.item() == 50256: # EOS
                break
                
            generated = torch.cat([generated, next_token], dim=1)
            
            word = tokenizer.decode(next_token[0])
            print(word, end="", flush=True)
            
    print("\n")

if __name__ == "__main__":
    main()
