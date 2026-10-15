import torch
import sys
import time
import math
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(str(Path(__file__).parent.parent))

from src.config import PCEIConfig
from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.core.ingestors import ImageIngestor, AudioIngestor
from src.tokenizer import PCEITokenizer
from src.engine.diffusion_utils import diffusion_training_step

console = Console()

def run_full_model_test():
    console.print(Panel.fit("[bold cyan]PCEI Full Model & Architecture Diagnostic Test[/bold cyan]"))
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[green]Device:[/green] {device}")
    
    config = PCEIConfig.load()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / "pcei_100m.pt"
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, config.layers_text, config.layers_img, config.layers_audio, config.d_model_text, config.d_model_img, config.d_model_audio
    )
    
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio
    
    console.print(f"Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | D_Text={d_text}")
    
    nexus = PCEINexus(config=config, device=str(device)).to(device)
    
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
        nexus.load_ambient_state(ckpt)
        console.print(f"[green]Loaded checkpoint: {ckpt_path.name}[/green]")
        
    tokenizer = PCEITokenizer()
    ingestor = ImageIngestor(dtype=nexus.system_dtype)
    audio_ingestor = AudioIngestor(dtype=nexus.system_dtype)
    
    results = {}
    
    # ── Test 1: Text Forward & Generation ──
    try:
        sample_prompt = "What is consciousness?"
        tokens = tokenizer.encode(sample_prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, t_lat, _, _, _, _ = nexus.core(
                text_input=tokens,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=0.5
            )
        assert logits is not None and not torch.isnan(logits).any()
        results["Text Stack Forward Pass"] = ("PASS", f"Output Shape: {list(logits.shape)}")
    except Exception as e:
        results["Text Stack Forward Pass"] = ("FAIL", str(e))
        
    # ── Test 2: Image Ingestion & Stack Forward ──
    try:
        dummy_img = torch.randn(1, 3, 224, 224, dtype=nexus.system_dtype, device=device)
        patches = dummy_img.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(1, -1, 3 * 16 * 16)
        with torch.no_grad():
            img_latent = nexus.core.img_stack.integrate_input(patches)
            _, _, _, _, img_recon, _ = nexus.core(
                img_latent=img_latent,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=0.5
            )
        assert img_recon is not None and not torch.isnan(img_recon).any()
        results["Image Stack Forward Pass"] = ("PASS", f"Recon Shape: {list(img_recon.shape)}")
    except Exception as e:
        results["Image Stack Forward Pass"] = ("FAIL", str(e))
        
    # ── Test 3: Audio Ingestion & Stack Forward ──
    try:
        dummy_audio = torch.randn(1, 100, 80, dtype=nexus.system_dtype, device=device)
        with torch.no_grad():
            audio_latent = nexus.core.audio_stack.integrate_input(dummy_audio)
            _, _, _, _, _, aud_recon = nexus.core(
                audio_latent=audio_latent,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=0.5
            )
        assert aud_recon is not None and not torch.isnan(aud_recon).any()
        results["Audio Stack Forward Pass"] = ("PASS", f"Recon Shape: {list(aud_recon.shape)}")
    except Exception as e:
        results["Audio Stack Forward Pass"] = ("FAIL", str(e))
        
    # ── Test 4: AmbientField Dual-Timescale State ──
    try:
        fast_norm = torch.norm(nexus.ambient_field.fast_state).item() if nexus.ambient_field.fast_state is not None else 0.0
        slow_norm = torch.norm(nexus.ambient_field.slow_state).item() if nexus.ambient_field.slow_state is not None else 0.0
        nexus.ambient_field.tick(t_lat, 0.5, detectors=nexus.core.detectors)
        nexus.ambient_field.tick_consolidation()
        new_fast_norm = torch.norm(nexus.ambient_field.fast_state).item()
        results["AmbientField Dual Timescale"] = ("PASS", f"Fast Norm: {fast_norm:.4f} -> {new_fast_norm:.4f}, Slow: {slow_norm:.4f}")
    except Exception as e:
        results["AmbientField Dual Timescale"] = ("FAIL", str(e))
        
    # ── Test 5: EntanglementBroker & Resonance Detectors ──
    try:
        bonds = nexus.core.broker.measure_entanglement(nexus.core.detectors)
        nexus.core.broker.tick_decay()
        results["Entanglement Physics"] = ("PASS", f"Active Bonds: {len(nexus.core.broker.active_bonds)}, Tracked Pairs: {len(nexus.core.detectors)}")
    except Exception as e:
        results["Entanglement Physics"] = ("FAIL", str(e))
        
    # ── Test 6: TopologyShifter & Dynamic Plasticity ──
    try:
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
        opt = torch.optim.AdamW(nexus.parameters(), lr=1e-4)
        nexus.adjust_plasticity(opt)
        eff_lr = opt.param_groups[0]['lr']
        results["Topology & Plasticity Modulation"] = ("PASS", f"Effective Base LR: {eff_lr:.2e}")
    except Exception as e:
        results["Topology & Plasticity Modulation"] = ("FAIL", str(e))
        
    # ── Test 7: Multimodal Cross-Modal Training Step ──
    try:
        dummy_text_tokens = tokenizer.encode("Sample test caption").unsqueeze(0).to(device)
        dummy_text_clean = nexus.core.text_stack.integrate_input(dummy_text_tokens)
        loss, mse_val, t, scale = diffusion_training_step(
            nexus, dummy_text_clean, max_T=5,
            ce_target=dummy_text_tokens, ce_weight=1.0,
            modality='text'
        )
        assert loss is not None and not torch.isnan(loss)
        results["Cross-Modal Diffusion Step"] = ("PASS", f"Step Loss: {loss.item():.4f} at t={t}")
    except Exception as e:
        results["Cross-Modal Diffusion Step"] = ("FAIL", str(e))

    table = Table(title="PCEI Architectural Diagnostics")
    table.add_column("Subsystem / Mechanism", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="magenta")
    
    for name, (status, detail) in results.items():
        status_styled = f"[green]{status}[/green]" if status == "PASS" else f"[red]{status}[/red]"
        table.add_row(name, status_styled, detail)
        
    console.print(table)

if __name__ == "__main__":
    run_full_model_test()
