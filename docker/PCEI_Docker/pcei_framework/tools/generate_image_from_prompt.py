import argparse
import torch
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.core.ingestors import ImageIngestor

class PCEITokenizer:
    def __init__(self, vocab_size: int = 50257):
        import tiktoken
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256
    
    def encode(self, text: str, pad: bool = True, max_len: int = 128) -> torch.Tensor:
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        ids = [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]
        if pad:
            ids = ids[:max_len]
            while len(ids) < max_len:
                ids.append(self.pad_id)
        return torch.tensor(ids, dtype=torch.long)

def get_mean_confidence(nexus: PCEINexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)

def save_tensor_as_image(tensor: torch.Tensor, filepath: Path):
    """Convert a (num_patches, patch_dim) latent back into a 32x32 image and save as PNG."""
    import numpy as np
    from PIL import Image
    img_np = tensor.detach().cpu().numpy()
    # Normalize
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    
    # Assume 4 patches of 16x16x3 (768) -> 32x32x3 grid
    grid = np.zeros((32, 32, 3))
    idx = 0
    for i in range(0, 32, 16):
        for j in range(0, 32, 16):
            if idx < img_np.shape[0]:
                grid[i:i+16, j:j+16, :] = img_np[idx].reshape((16, 16, 3))
            idx += 1
            
    grid = (grid * 255).astype(np.uint8)
    img = Image.fromarray(grid)
    img.resize((256, 256), Image.NEAREST).save(filepath)

def main():
    parser = argparse.ArgumentParser(description="Generate a single image from a text prompt using the PCEI model.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired image.")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_500m.pt", help="Checkpoint file in checkpoints/.")
    parser.add_argument("--output", type=str, default="artifacts/generated_image.png", help="Where to write the PNG.")
    args = parser.parse_args()

    console = Console()
    console.print(Panel.fit("[bold magenta]PCEI Prompt-to-Image Generator[/bold magenta]\n[yellow]Generating...[/yellow]"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    console.print(f"[cyan]Using device: {device}[/cyan]")

    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    # Load model topology from checkpoint
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        checkpoint_path, 2, 1, 1
    )
    console.print(f"[cyan]Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio[/cyan]")
    console.print(f"[cyan]Dimensions: {d_text} Text | {d_img} Img | {d_audio} Audio[/cyan]")

    nexus = PCEINexus(
        vocab_size=50257,
        device=device,
        d_model_text=d_text,
        d_model_img=d_img,
        d_model_audio=d_audio,
        layers_text=layers_text,
        layers_img=layers_img,
        layers_audio=layers_audio
    ).to(device)

    if checkpoint_path.exists():
        console.print(f"[green]Loading weights from {checkpoint_path}...[/green]")
        ckpt = torch.load(checkpoint_path, map_location=device)
        nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        console.print(f"[red]Checkpoint {checkpoint_path} not found! Generating with random weights.[/red]")

    tokenizer = PCEITokenizer()
    text_tokens = tokenizer.encode(args.prompt, pad=True).unsqueeze(0).to(device)
    
    with torch.no_grad():
        text_latent = nexus.core.text_stack.integrate_input(text_tokens)
        
    # Start with noise in the image latent space (using d_model_img)
    # Using a slightly larger scale for noise
    img_latent = torch.randn(1, 4, d_img, device=device) * 0.5
    
    # Run multiple steps of iterative refinement
    steps = 25
    console.print(f"[yellow]Running {steps} diffusion steps...[/yellow]")
    
    with torch.no_grad():
        for i in range(steps):
            # We use t/steps as the time embedding
            t_val = (steps - i - 1) / steps
            t_tensor = torch.tensor([t_val], device=device)
            
            # The 3rd return value is the updated i_latent
            results = nexus.core(
                text_latent=text_latent,
                img_latent=img_latent,
                diffusion_timesteps=1, # One tick of the internal loop
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=get_mean_confidence(nexus)
            )
            img_latent = results[2] # i_latent
            
            # Optional: Add a tiny bit of noise back in (Langevin dynamics)
            if i < steps - 1:
                img_latent = img_latent + torch.randn_like(img_latent) * 0.01

        # Final decode to pixels
        img_recon = nexus.core.img_stack.decode_output(img_latent)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor_as_image(img_recon[0], out_path)
    console.print(f"[bold green]Success![/bold green] Image saved to [cyan]{out_path}[/cyan]")

if __name__ == "__main__":
    main()
