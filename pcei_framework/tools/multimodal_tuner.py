import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
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

from src.tokenizer import PCEITokenizer
from dashboard.telemetry import PCEITelemetry

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
    tensor = tensor.float()
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
    """Atomic save with backup using the Golden Rule."""
    save_dict = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    from src.pcei import PCEINexus
    if PCEINexus.save_safe_checkpoint(nexus, save_dict, checkpoint_path, create_backup=True):
        print(f"[green]Checkpoint solidified at {checkpoint_path.name}[/green]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCEI Multimodal Tuner")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--use-metabolic-loss", action="store_true", default=False)
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--audio-only", action="store_true", help="Only train audio-text bonds")
    parser.add_argument("--image-only", action="store_true", help="Only train image-text bonds")
    parser.add_argument("--metabolic-flush", action="store_true", default=False, help="Reset slow-field confidence to zero before training")
    args = parser.parse_args()

    from src.config import PCEIConfig
    config = PCEIConfig.load()

    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name

    console.print(Panel.fit("[bold magenta]PCEI Multimodal Tuner[/bold magenta]\n[yellow]Continuous Fine-Tuning for Sights & Sounds[/yellow]"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    console.print(f"[cyan]Using device: {device}[/cyan]")
    
    tokenizer = PCEITokenizer()
    # Will be reinitialized with proper dtype after loading nexus
    ingestor = None
    audio_ingestor = None
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path if "ckpt_path" in locals() else checkpoint_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    console.print(f"[cyan]Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text[/cyan]")
    
    nexus = PCEINexus(
        config=config,
        device=device
    ).to(device)
    
    ingestor = ImageIngestor(dtype=nexus.system_dtype)
    audio_ingestor = AudioIngestor(dtype=nexus.system_dtype)

    
    optimizer = AdamW(nexus.parameters(), lr=args.lr, weight_decay=0.01)
    
    if checkpoint_path.exists():
        console.print(f"[green]Loading checkpoint from {checkpoint_path.name}...[/green]")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
            nexus.load_ambient_state(ckpt)
            nexus.to(nexus.system_dtype)  # Re-enforce dtype — load_state_dict can leave stray float32 submodules
            
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
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    # Force all optimizer states to bfloat16 for MPS compatibility
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(nexus.system_dtype)
                except: pass
            console.print("[green]Weights loaded successfully.[/green]")
        except Exception as e:
            console.print(f"[bold red]CRITICAL: Checkpoint exists but is unreadable: {e}[/bold red]")
            console.print("[red]Aborting to prevent overwriting with Step 0 model.[/red]")
            sys.exit(1)
    else:
        console.print(f"[yellow]Initial pcei_100m.pt not found. Starting new foundation.[/yellow]")
        
    if args.metabolic_flush:
        console.print("[System] Metabolic Flush: Resetting slow-field confidence to zero...")
        with torch.no_grad():
            for m in nexus.core.modules():
                if isinstance(m, PlasticLayer):
                    m.slow_field_confidence.zero_()
        
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
            if telem.shutdown_requested:
                break
            mode = random.choice(available_modes)
            print(f"DEBUG: Selected mode: {mode}")

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

                # Pad/trim all to same length dynamically based on the longest clip in the batch
                max_t = max(a.shape[0] for a in target_audio_list)
                padded = []
                for a in target_audio_list:
                    if a.shape[0] >= max_t:
                        padded.append(a[:max_t, :])
                    else:
                        pad = torch.zeros(max_t - a.shape[0], a.shape[1], dtype=a.dtype)
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
                    prompts = [f"USER: Describe this image.\nASSISTANT: {cap}" for cap in valid_captions]
                    text_tokens = torch.stack([tokenizer.encode(p, pad=True) for p in prompts]).to(device)
                    if text_tokens.size(1) > 512: text_tokens = text_tokens[:, :512]

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
                    prompts = [f"USER: Generate an image of: {cap}\nASSISTANT:" for cap in valid_captions]
                    text_tokens = torch.stack([tokenizer.encode(p, pad=True) for p in prompts]).to(device)
                    if text_tokens.size(1) > 512: text_tokens = text_tokens[:, :512]

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
                                override_timestep=0,
                                slow_state=nexus.ambient_field.slow_state,
                                mean_confidence=get_mean_confidence(nexus)
                            )
                            if img_recon is not None:
                                # Cast to float32 for torchvision compatibility
                                save_tensor_as_image(img_recon[0].float(), vis_path)

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
            nexus.neurogenesis.monitor_gradients()
            
            # --- GRADIENT ACCUMULATION & PHYSICS TICK ---
            if (step + 1) % args.grad_accum_steps == 0:
                # --- SCORCH EARTH GRADIENT SANITIZER ---
                for p in nexus.parameters():
                            pass # nan_to_num band-aid removed
                        
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
                            spawned_layers = nexus.neurogenesis.apply_structural_changes(structural_actions)
                            if spawned_layers:
                                for layer in spawned_layers:
                                    optimizer.add_param_group({'params': layer.parameters()})
                            nexus.core.shifter.refresh_registry()
        
            console.print(f"Step {step:03d} | Mode: {mode:10s} | Loss: {loss_val:.4f} | Bonds: {len(nexus.core.broker.active_bonds)}")
            if not sys.stdout.isatty():
                print(f"[Step {step}] Loss: {loss_val:.4f} | Active Bonds: {len(nexus.core.broker.active_bonds)} | LR: 4.00e-04", flush=True)
            
            telem.emit(
                step=step,
                loss=loss_val,
                active_bonds=len(nexus.core.broker.active_bonds),
                lr=optimizer.param_groups[0]['lr'],
                mean_confidence=get_mean_confidence(nexus)
            )

            if 'mps' in str(device):
                torch.mps.empty_cache()
            
            if step > 0 and step % 100 == 0:
                save_checkpoint(nexus, optimizer, checkpoint_path)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupt received. Saving current physics state...[/bold red]")
        save_checkpoint(nexus, optimizer, checkpoint_path)
        console.print("[green]Physical state calcified. Safe to exit.[/green]")
    finally:
        telem.close()

if __name__ == "__main__":
    import subprocess, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    main()
