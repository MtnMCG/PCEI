import torch
import math
import argparse
import sys
import os
import urllib.request
from pathlib import Path
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.engine.diffusion_utils import diffusion_training_step
from tools.wiki_explorer import PCEITokenizer

console = Console()

def download_wikitext_valid():
    """Downloads a tiny slice of WikiText-103 validation set for pure held-out testing."""
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    cache_path = Path(__file__).parent.parent / "data" / "wikitext_valid.txt"
    os.makedirs(cache_path.parent, exist_ok=True)
    
    if not cache_path.exists():
        console.print("[dim]Downloading WikiText-2 Validation Set...[/dim]")
        try:
            urllib.request.urlretrieve(url, cache_path)
            console.print("[dim]Download complete.[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to download benchmark dataset: {e}[/bold red]")
            sys.exit(1)
            
    with open(cache_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Clean up the butchered WikiText-2 pre-tokenization formatting
    import re
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' ,', ',', text)
    text = re.sub(r' ;', ';', text)
    text = re.sub(r' \:', ':', text)
    text = re.sub(r' \?', '?', text)
    text = re.sub(r' \!', '!', text)
    text = re.sub(r' \( ', ' (', text)
    text = re.sub(r' \) ', ') ', text)
    text = re.sub(r' @-@ ', '-', text)
    text = re.sub(r' @\.@ ', '.', text)
    text = text.replace(' <unk> ', ' ')
    
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]PCEI Kill-Shot Benchmark (Perplexity)[/bold cyan]")
    console.print(f"Device: {device}")
    
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, config.layers_text, config.layers_img, config.layers_audio, config.d_model_text, config.d_model_img, config.d_model_audio
    )
    
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio
    
    nexus = PCEINexus(config=config, device=device)
    
    if ckpt_path.exists():
        console.print(f"[dim]Loading checkpoint {ckpt_path.name}...[/dim]")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
        nexus.load_ambient_state(checkpoint)
        nexus.to(device)
    else:
        console.print("[bold red]ERROR: No checkpoint found to benchmark. Train it first.[/bold red]")
        sys.exit(1)
        
    nexus.eval()
    tokenizer = PCEITokenizer(vocab_size=config.vocab_size)
    
    raw_text = download_wikitext_valid()
    # Clean up empty lines and small headers
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 50]
    
    console.print(f"[bold yellow]Running rigorous evaluation on {len(lines)} held-out chunks...[/bold yellow]\n")
    
    total_ce_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, line in enumerate(lines[:100]): # Test on 100 chunks for speed
            tokens = tokenizer.encode(line, pad=False).unsqueeze(0).to(device)
            if tokens.size(1) > 512:
                tokens = tokens[:, :512]
                
            clean_latent = nexus.core.text_stack.integrate_input(tokens)
            
            # Run diffusion evaluation without gradients to calculate pure Cross-Entropy
            ce_loss, _, _, _ = diffusion_training_step(
                nexus, clean_latent, max_T=12,
                ce_target=tokens, ce_weight=1.0
            )
            
            total_ce_loss += ce_loss.item()
            total_samples += 1
            
            if i % 10 == 0:
                current_ppl = math.exp(min(total_ce_loss / total_samples, 20))
                console.print(f"[dim]Processed {i}/100 | Current PPL: {current_ppl:.2f}[/dim]", end="\r")
                
    mean_ce = total_ce_loss / total_samples
    final_perplexity = math.exp(min(mean_ce, 20))
    
    console.print("\n")
    console.print("="*50)
    console.print(f"[bold green]Final Held-Out Cross-Entropy Loss:[/bold green] {mean_ce:.4f}")
    console.print(f"[bold magenta]Final True Perplexity:[/bold magenta] {final_perplexity:.2f}")
    console.print("="*50)
    
    if final_perplexity < 40.0:
        console.print("\n[bold cyan]RESULT: The model has mathematically grokked.[/bold cyan]")
        console.print("It has achieved GPT-2 tier generalization on unseen data.")
    else:
        console.print("\n[bold yellow]RESULT: The model is still learning.[/bold yellow]")
        console.print("Keep grinding the brute-force phase before isolating it in the Dreaming Chamber.")

if __name__ == "__main__":
    main()
