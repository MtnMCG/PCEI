import argparse
import torch
import sys
from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.panel import Panel

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus
from src.core.ingestors import ImageIngestor

class PCEITokenizer:
    def __init__(self, vocab_size: int = 50257):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256
    
    def encode(self, text: str, pad: bool = True, max_len: int = 128) -> torch.Tensor:
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        if pad:
            ids = raw_ids[:max_len]
            while len(ids) < max_len:
                ids.append(self.pad_id)
        else:
            ids = raw_ids
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        # Filter padding
        clean_ids = [i for i in ids.tolist() if i != self.pad_id]
        return self.enc.decode(clean_ids)

def main():
    parser = argparse.ArgumentParser(description="Test PCEI Image Captioning.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_500m.pt")
    args = parser.parse_args()

    console = Console()
    console.print(Panel.fit("[bold magenta]PCEI Image Captioning Test[/bold magenta]"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    # Load topology
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(checkpoint_path, 2, 1, 1)
    nexus = PCEINexus(vocab_size=50257, device=device, d_model_text=d_text, d_model_img=d_img, d_model_audio=d_audio,
                      layers_text=layers_text, layers_img=layers_img, layers_audio=layers_audio).to(device)
    
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
        console.print(f"[green]Loaded {checkpoint_path.name}[/green]")
    
    tokenizer = PCEITokenizer()
    ingestor = ImageIngestor()
    
    # Load and process image
    img_path = Path(args.image)
    if not img_path.exists():
        console.print(f"[red]Image not found: {img_path}[/red]")
        return
        
    img_tensor = ingestor.load_image(img_path).to(device) # (1, num_patches, patch_dim)
    
    # Prefix for captioning
    prefix_text = "USER: Describe this image.\nBEAST:"
    tokens = tokenizer.encode(prefix_text, pad=False).unsqueeze(0).to(device)
    
    # Generate completion
    generated = tokens
    console.print(f"[cyan]Input Image: {img_path.name}[/cyan]")
    console.print(f"[yellow]Generating caption...[/yellow]")
    
    for _ in range(20):
        with torch.no_grad():
            # Integrate current text prefix
            text_latent = nexus.core.text_stack.integrate_input(generated)
            
            # Forward pass through core with image context
            # We want to denoise the text latent using the image as context
            _, denoised_text, _, _, _, _ = nexus.core(
                text_latent=text_latent,
                img_input=img_tensor,
                diffusion_timesteps=1,
                slow_state=nexus.ambient_field.slow_state
            )
            
            # Get logits for the LAST token
            logits = nexus.core.text_stack.decode_output(denoised_text)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS or newline?
            if next_token.item() == tokenizer.eos_id:
                break
                
    full_text = tokenizer.decode(generated[0])
    console.print(f"[bold green]Result:[/bold green] {full_text}")

if __name__ == "__main__":
    main()
