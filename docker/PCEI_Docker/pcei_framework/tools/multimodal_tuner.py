import os
import sys
import math
import time
import json
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from rich.console import Console
from rich.panel import Panel

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.core.ingestors import ImageIngestor, AudioIngestor
from src.engine.diffusion_utils import diffusion_training_step

# Base project directory for relative paths
BASE = Path(__file__).parent.parent

console = Console()

# Checkpoint constants removed in favor of argparse
DATA_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "massive_data_test_manifest.json"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "multimodal_tuner"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
    
    def decode(self, ids: torch.Tensor) -> str:
        raw_ids = [tok_id.item() for tok_id in ids if tok_id.item() != self.pad_id]
        raw_ids = [0 if tok_id == (self.vocab_size - 1) else tok_id for tok_id in raw_ids]
        try:
            return self.enc.decode(raw_ids)
        except Exception:
            return "<decode_error>"

def get_mean_confidence(nexus: PCEINexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)

def load_dataset(manifest_path: Path):
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found at {manifest_path}[/red]")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    image_records = [d for d in data if d.get('modality') == 'image']
    audio_records = [d for d in data if d.get('modality') == 'audio']
    console.print(f"  [dim]Image records: {len(image_records)} | Audio records: {len(audio_records)}[/dim]")
    return image_records, audio_records

def save_tensor_as_image(tensor: torch.Tensor, filepath: Path):
    """Converts a (num_patches, patch_dim) latent back into an image (for visual debugging)."""
    # Inverse of the ImageIngestor standard
    try:
        # Patch_dim = 16x16x3 = 768. Target = 32x32. Num_patches = 4
        # Assuming (4, 768)
        img_np = tensor.detach().cpu().numpy()
        # Restore some variance for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
        
        # Reshape to 2x2 grid of 16x16x3 patches -> 32x32x3
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
    except Exception as e:
        console.print(f"[red]Failed to save image visualization: {e}[/red]")

def save_checkpoint(nexus, optimizer, checkpoint_path):
    """Atomic save with backup."""
    if checkpoint_path.exists():
        backup_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_BACKUP.pt")
        shutil.copy2(checkpoint_path, backup_path)
        
    torch.save({
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(checkpoint_path) + '.tmp')
    os.replace(str(checkpoint_path) + '.tmp', str(checkpoint_path))
    print(f"[green]Checkpoint solidified at {checkpoint_path.name}[/green]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCEI Multimodal Tuner")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--use-metabolic-loss", action="store_true", default=False)
    parser.add_argument("--checkpoint-name", type=str, default="pcei_micro.pt")
    parser.add_argument("--audio-only", action="store_true", help="Only train audio-text bonds")
    parser.add_argument("--image-only", action="store_true", help="Only train image-text bonds")
    args = parser.parse_args()

    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name

    console.print(Panel.fit("[bold magenta]PCEI Multimodal Tuner[/bold magenta]\n[yellow]Continuous Fine-Tuning for Sights & Sounds[/yellow]"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    console.print(f"[cyan]Using device: {device}[/cyan]")
    
    tokenizer = PCEITokenizer()
    ingestor = ImageIngestor()
    audio_ingestor = AudioIngestor()
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(checkpoint_path, 2, 1, 1)
    console.print(f"[cyan]Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text[/cyan]")
    
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
    
    optimizer = AdamW(nexus.parameters(), lr=args.lr, weight_decay=0.01)
    
    if checkpoint_path.exists():
        console.print(f"[green]Loading checkpoint from {checkpoint_path.name}...[/green]")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
            
            # --- WEIGHT SANITIZER & DEFIBRILLATOR ---
            with torch.no_grad():
                nan_count = 0
                revived_count = 0
                for name, p in nexus.named_parameters():
                    # 1. Scrub NaNs/Infs
                    mask = torch.isnan(p) | torch.isinf(p)
                    if mask.any():
                        nan_count += mask.sum().item()
                        p.data[mask] = 0.0
                    
                    # 2. Defibrillator: Detect Dead Layers (All Zero)
                    if "weight" in name and p.numel() > 10:
                        zero_ratio = (p.data == 0).float().mean().item()
                        if zero_ratio > 0.90:
                            revived_count += 1
                            if p.dim() >= 2:
                                nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                            else:
                                nn.init.normal_(p.data, std=0.02)
                
                if nan_count > 0:
                    console.print(f"[yellow]Sanitizer: Cleaned {nan_count} corrupted weights.[/yellow]")
                if revived_count > 0:
                    console.print(f"[cyan]Defibrillator: Revived {revived_count} collapsed layers.[/cyan]")

            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: pass
            console.print("[green]Weights loaded successfully.[/green]")
        except Exception as e:
            console.print(f"[bold red]CRITICAL: Checkpoint exists but is unreadable: {e}[/bold red]")
            console.print("[red]Aborting to prevent overwriting with Step 0 model.[/red]")
            sys.exit(1)
    else:
        console.print(f"[yellow]Initial pcei_micro.pt not found. Starting new foundation.[/yellow]")
        
    image_records, audio_records = load_dataset(DATA_MANIFEST_PATH)
    console.print(f"[cyan]Loaded {len(image_records)} image + {len(audio_records)} audio records.[/cyan]")

    # Decide available training modes based on what data we have
    available_modes = []
    if image_records and not args.audio_only:
        available_modes += ["CAPTIONING", "GENERATION"]
    if audio_records and not args.image_only:
        available_modes += ["AUDIO_SPEECH"]
    if not available_modes:
        console.print("[red]No training data found in manifest for chosen modalities.[/red]")
        sys.exit(1)
    console.print(f"[cyan]Training modes active: {available_modes}[/cyan]")
    
    nexus.set_mode("ACTIVE")
    
    diffusion_steps = 3
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    mse_loss = nn.MSELoss()
    
    optimizer.zero_grad()
    
    try:
        for step in range(args.steps):
            mode = random.choice(available_modes)

            # ── AUDIO_SPEECH MODE ─────────────────────────────────────────────
            if mode == "AUDIO_SPEECH":
                batch_records = random.choices(audio_records, k=args.batch_size)
                target_audio_list = []
                valid_transcripts = []

                for r in batch_records:
                    aud_path = BASE / r.get('file', '')
                    if not aud_path.exists():
                        continue
                    try:
                        # (1, T, mel_bins) → squeeze to (T, mel_bins)
                        aud_tensor = audio_ingestor.load_audio(aud_path).squeeze(0)
                        target_audio_list.append(aud_tensor)
                        valid_transcripts.append(r.get('caption', ''))
                    except Exception as e:
                        continue

                if not target_audio_list:
                    continue

                # Pad/trim all to same length
                max_t = min(max(a.shape[0] for a in target_audio_list), 256)
                padded = []
                for a in target_audio_list:
                    if a.shape[0] >= max_t:
                        padded.append(a[:max_t, :])
                    else:
                        pad = torch.zeros(max_t - a.shape[0], a.shape[1])
                        padded.append(torch.cat([a, pad], dim=0))
                target_audio = torch.stack(padded).to(device)  # (B, T, mel_bins)

                # Encode transcripts as text context
                text_tokens = torch.stack(
                    [tokenizer.encode(t, pad=True) for t in valid_transcripts]
                ).to(device)

                with torch.no_grad():
                    text_latent = nexus.core.text_stack.integrate_input(text_tokens)
                    audio_clean = nexus.core.audio_stack.integrate_input(target_audio)

                # Add noise to audio latent — model denoises using text as context
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, audio_clean, max_T=12,
                    text_latent=text_latent,
                    modality='audio'
                )

                task_loss = total_loss

            # ── IMAGE MODES ───────────────────────────────────────────────────
            else:
                # Image mode – sample a batch of image records
                batch_records = random.choices(image_records, k=args.batch_size)
                target_images = []
                valid_captions = []

                for r in batch_records:
                    img_full_path = Path(__file__).parent.parent / r.get('file', '')
                    if img_full_path.exists():
                        try:
                            img_tensor = ingestor.load_image(img_full_path)
                            target_images.append(img_tensor.squeeze(0))
                            valid_captions.append(r.get('caption', 'an image'))
                        except Exception:
                            pass

                if not target_images:
                    continue

                target_images = torch.stack(target_images).to(device)  # (B, num_patches, patch_dim)

                if mode == "CAPTIONING":
                    # Image (Clean) -> Text (Noisy Target)
                    prompts = [f"USER: Describe this image.\nBEAST: {cap}" for cap in valid_captions]
                    text_tokens = torch.stack([tokenizer.encode(p, pad=True) for p in prompts]).to(device)

                    with torch.no_grad():
                        text_clean = nexus.core.text_stack.integrate_input(text_tokens)
                        img_clean = nexus.core.img_stack.integrate_input(target_images)

                    # Diffusion training: denoise text with image as context
                    task_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                        nexus, text_clean, max_T=12,
                        ce_target=text_tokens, ce_weight=0.3,
                        img_latent=img_clean,
                        modality='text'
                    )

                else:  # GENERATION
                    # Text (Clean) -> Image (Noisy Target)
                    prompts = [f"USER: Generate an image of: {cap}\nBEAST:" for cap in valid_captions]
                    text_tokens = torch.stack([tokenizer.encode(p, pad=True) for p in prompts]).to(device)

                    with torch.no_grad():
                        text_latent = nexus.core.text_stack.integrate_input(text_tokens)
                        img_clean = nexus.core.img_stack.integrate_input(target_images)

                    # Diffusion training: denoise image with text as context
                    task_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                        nexus, img_clean, max_T=12,
                        text_latent=text_latent,
                        modality='img'
                    )
                    
                    # Save visualization occasionally
                    if step % 50 == 0:
                        vis_path = ARTIFACTS_DIR / f"recon_step_{step}.png"
                        # Use a quick forward pass for visualization
                        with torch.no_grad():
                            _, _, _, _, img_recon, _ = nexus.core(
                                text_latent=text_latent,
                                img_latent=img_clean,
                                diffusion_timesteps=1,
                                slow_state=nexus.ambient_field.slow_state,
                                mean_confidence=get_mean_confidence(nexus)
                            )
                            if img_recon is not None:
                                save_tensor_as_image(img_recon[0], vis_path)

            # end image modes
            
            # --- METABOLIC LOSS (Physics Penalty) ---
            total_loss = task_loss
            if args.use_metabolic_loss and hasattr(nexus.core, 'shifter'):
                # The shifter calculates proximity reward based on active bonds
                # We want to maximize reward, so minimize -reward
                try:
                    metabolic_reward = nexus.core.shifter.calculate_proximity_reward(nexus.core.broker.active_bonds)
                    metabolic_loss = -metabolic_reward * 0.01  # Small penalty weight
                    total_loss = task_loss + metabolic_loss
                except Exception:
                    pass
                    
            # Scale for gradient accumulation
            loss_val = total_loss.item()
            total_loss = total_loss / args.grad_accum_steps
            total_loss.backward()
            
            # --- GRADIENT ACCUMULATION & PHYSICS TICK ---
            if (step + 1) % args.grad_accum_steps == 0:
                # --- SCORCH EARTH GRADIENT SANITIZER ---
                for p in nexus.parameters():
                    if p.grad is not None:
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0)
                        
                torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Physics Cycle
                if hasattr(nexus.core, 'shifter'):
                    nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
                if hasattr(nexus.core, 'broker'):
                    nexus.core.broker.tick_decay()
                    
                if hasattr(nexus, 'ambient_field'):
                    nexus.ambient_field.tick_consolidation()
                    
                if hasattr(nexus, 'neurogenesis'):
                    if step % 20 == 0: # Check less often due to batches
                        structural_actions = nexus.neurogenesis.evaluate_topology()
                        if structural_actions:
                            nexus.neurogenesis.apply_structural_changes(structural_actions)
                            nexus.core.shifter.refresh_registry()
        
            console.print(f"Step {step:03d} | Mode: {mode:10s} | Loss: {loss_val:.4f} | Bonds: {len(nexus.core.broker.active_bonds)}")
            
            if step > 0 and step % 100 == 0:
                save_checkpoint(nexus, optimizer, checkpoint_path)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupt received. Saving current physics state...[/bold red]")
        save_checkpoint(nexus, optimizer, checkpoint_path)
        console.print("[green]Physical state calcified. Safe to exit.[/green]")

if __name__ == "__main__":
    import subprocess, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    main()
