import torch
import torch.nn as nn
import math
import argparse
import sys
import os
import time
import random
from pathlib import Path
import tiktoken

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step

console = Console()

class PCEITokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256

    def encode(self, text: str, max_len: int = 64, pad: bool = True) -> torch.Tensor:
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


class DreamingUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="log", size=10)
        )
        self.layout["main"].split_row(
            Layout(name="dream_stream", ratio=2),
            Layout(name="metrics", ratio=1)
        )

        self.dream_cycle = 0
        self.seed_topic = "None (Pure Noise)"
        self.last_dream_text = ""
        self.dream_history = []
        self.metrics = {
            "mean_confidence": 0.0,
            "gradient_norm": 0.0,
            "dream_strength": 0.0,
            "bonds": 0,
            "bonds_formed_total": 0,
            "bonds_dissolved_total": 0,
            "dream_loss": 0.0,
            "consolidation_loss": 0.0,
            "layers_alive": 0,
            "topology_spread": 0.0,
        }
        self.logs = []

    def log(self, message: str):
        time_str = time.strftime("%H:%M:%S")
        log_line = f"[{time_str}] {message}"
        self.logs.append(log_line)
        if len(self.logs) > 8:
            self.logs.pop(0)

        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "dreaming.log", "a") as f:
            f.write(log_line + "\n")

    def update_layout(self):
        # Header
        self.layout["header"].update(Panel(
            f"[bold cyan]PCEI Dreaming Chamber[/bold cyan] — Cycle [bold yellow]{self.dream_cycle}[/bold yellow] | Seed: [bold green]{self.seed_topic}[/bold green]",
            style="white on blue"
        ))

        # Dream Stream — show recent dream fragments
        dream_lines = self.dream_history[-12:]
        dream_display = Text()
        for i, line in enumerate(dream_lines):
            opacity = "dim" if i < len(dream_lines) - 3 else ""
            dream_display.append(f"  [{i+1:03d}] {line}\n", style=opacity)

        self.layout["dream_stream"].update(Panel(
            dream_display,
            title="Internal Dialogue Stream",
            border_style="magenta"
        ))

        # Metrics
        m_table = Table.grid(padding=1)
        m_table.add_column(style="cyan", justify="right")
        m_table.add_column(style="magenta")
        m_table.add_row("Dream Cycle:", f"{self.dream_cycle}")
        m_table.add_row("─" * 20, "─" * 10)
        m_table.add_row("Mean Confidence:", f"{self.metrics['mean_confidence']:.4f}")
        m_table.add_row("Dream Strength:", f"{self.metrics['dream_strength']:.6f}")
        m_table.add_row("Dream Loss:", f"{self.metrics['dream_loss']:.4f}")
        m_table.add_row("Consolidation Loss:", f"{self.metrics['consolidation_loss']:.4f}")
        m_table.add_row("─" * 20, "─" * 10)
        m_table.add_row("Active Bonds:", f"{self.metrics['bonds']}")
        m_table.add_row("Bonds Formed:", f"{self.metrics['bonds_formed_total']}")
        m_table.add_row("Bonds Dissolved:", f"{self.metrics['bonds_dissolved_total']}")
        m_table.add_row("─" * 20, "─" * 10)
        m_table.add_row("Gradient Norm:", f"{self.metrics['gradient_norm']:.4f}")
        m_table.add_row("Layers Alive:", f"{self.metrics['layers_alive']}")
        m_table.add_row("Topology Spread:", f"{self.metrics['topology_spread']:.4f}")

        self.layout["metrics"].update(Panel(m_table, title="Physics Engine", border_style="green"))

        # Logs
        log_text = Text("\n".join(self.logs))
        self.layout["log"].update(Panel(log_text, title="System Log", border_style="yellow"))

        return self.layout


def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)


def get_topology_stats(nexus):
    positions = []
    count = 0
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            pos = module.get_position()
            positions.append(pos)
            count += 1

    if count < 2:
        return count, 0.0

    # Calculate spread as standard deviation of positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    spread = (torch.tensor(xs).std().item() + torch.tensor(ys).std().item()) / 2
    return count, spread


def compute_grad_norm(nexus) -> float:
    total_norm = 0.0
    for p in nexus.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def main(args):
    ui = DreamingUI()
    console.clear()

    # Prompt for optional seed topic
    seed_input = console.input(
        "[bold cyan]Enter a seed topic to prime the dreams (or press Enter for pure self-resonance): [/bold cyan]"
    )

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = PCEITokenizer()

    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, args.layers_text, args.layers_img, args.layers_audio
    )
    ui.log(f"Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")

    nexus = PCEINexus(
        vocab_size=50257,
        device=device,
        d_model_text=d_text,
        d_model_img=d_img,
        d_model_audio=d_audio,
        layers_text=layers_text,
        layers_img=layers_img,
        layers_audio=layers_audio
    )

    optimizer = torch.optim.AdamW(nexus.parameters(), lr=args.lr)
    if ckpt_path.exists():
        ui.log(f"Loading checkpoint: {ckpt_path.name}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
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
                    ui.log(f"Sanitizer: Cleaned {nan_count} corrupted weights.")
                if revived_count > 0:
                    ui.log(f"Defibrillator: Revived {revived_count} collapsed layers.")
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    pass
            ui.log("Foundation loaded successfully.")
        except Exception as e:
            ui.log(f"CRITICAL: Failed to load: {e}")
            ui.log("ABORTING to prevent wipe.")
            sys.exit(1)
    else:
        ui.log("No checkpoint found. Dreaming from blank noise.")

    # Seed priming: if user gave a topic, run one witness pass on it
    seed_latent = None
    if seed_input.strip():
        ui.seed_topic = seed_input.strip()
        ui.log(f"Priming dream seed: '{ui.seed_topic}'")
        seed_tokens = tokenizer.encode(seed_input.strip(), max_len=64).unsqueeze(0).to(device)
        with torch.no_grad():
            _, seed_latent, _, _, _, _ = nexus.core(
                text_input=seed_tokens,
                diffusion_timesteps=3,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=get_mean_confidence(nexus)
            )
        nexus.ambient_field.tick(seed_latent, 0.0, detectors=nexus.core.detectors)
        ui.log("Seed absorbed into AmbientField.")
    else:
        ui.seed_topic = "None (Pure Self-Resonance)"

    nexus.train()

    with Live(ui.update_layout(), console=console, screen=True, refresh_per_second=2) as live:
        cycle = 0
        while True:
            try:
                cycle += 1
                ui.dream_cycle = cycle
                mean_conf = get_mean_confidence(nexus)

                # ═══════════════════════════════════════
                # PHASE 1: INTERNAL DIALOGUE (Dreaming)
                # ═══════════════════════════════════════
                # The DreamSeeder generates noise shaped by the AmbientField.
                # The model runs a denoising pass on this noise with NO external input.
                optimizer.zero_grad()

                # Generate dream noise from AmbientField
                batch_size = 1
                intent = torch.zeros(batch_size, d_text, device=device)

                # If we have a seed latent, mix it in (decaying over time)
                if seed_latent is not None:
                    decay_factor = max(0.01, 1.0 - (cycle / 200.0))
                    intent = intent + seed_latent.mean(dim=1) * decay_factor
                
                # Anchor the dream to the intent/memory
                nexus.core.dream_seeder.prime(intent)

                # Build dream target from slow_state (what the model has consolidated)
                if nexus.ambient_field.slow_state is not None:
                    target_state = nexus.ambient_field.slow_state.detach().to(device)
                    dream_target = target_state.view(1, 1, -1).expand(1, 64, -1)
                else:
                    # No slow_state yet — use intent as a weak target
                    dream_target = intent.unsqueeze(1).expand(1, 64, -1)
                
                # Diffusion training: sample random timestep, denoise toward dream target
                total_loss_val, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, dream_target, max_T=12
                )
                
                total_loss_val.backward()
                total_loss = total_loss_val.item()
                
                # Get dream output for decoding
                with torch.no_grad():
                    dream_output = nexus.core.text_stack.integrate_input(
                        torch.zeros(1, 64, dtype=torch.long, device=device)
                    ) if False else dream_target  # Use target as proxy
                    # Actually generate dream output via a quick forward pass
                    _, dream_output, _, _, _, _ = nexus.core(
                        text_latent=dream_target,
                        diffusion_timesteps=1,
                        slow_state=nexus.ambient_field.slow_state,
                        mean_confidence=mean_conf
                    )
                
                nexus.core.dream_seeder.clear()
                
                # --- SCORCH EARTH GRADIENT SANITIZER ---
                torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
                grad_norm = compute_grad_norm(nexus)
                optimizer.step()

                # ═══════════════════════════════════════
                # PHASE 2: DECODE DREAM (What did it think?)
                # ═══════════════════════════════════════
                with torch.no_grad():
                    dream_logits = nexus.core.text_stack.decode_output(dream_output)
                    probs = torch.softmax(dream_logits[0] / 0.8, dim=-1)
                    dream_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    # Only decode first 20 tokens for display
                    dream_text = tokenizer.decode(dream_tokens[:20])

                ui.dream_history.append(dream_text.strip()[:120])
                if len(ui.dream_history) > 50:
                    ui.dream_history.pop(0)

                # ═══════════════════════════════════════
                # PHASE 3: PHYSICS TICK
                # ═══════════════════════════════════════
                # Consolidation
                nexus.ambient_field.tick(dream_output, total_loss / 20.0, detectors=nexus.core.detectors)

                # Resonance & Entanglement
                bonds = nexus.core.broker.measure_entanglement(nexus.core.detectors)
                nexus.core.broker.tick_decay()

                # Topology migration
                nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)

                # Neurogenesis check (every 10 cycles)
                if cycle % 10 == 0:
                    structural_actions = nexus.neurogenesis.evaluate_topology()
                    if structural_actions:
                        nexus.neurogenesis.apply_structural_changes(structural_actions)
                        nexus.core.shifter.refresh_registry()
                        ui.log(f"Neurogenesis event: {len(structural_actions)} structural changes")

                # ═══════════════════════════════════════
                # PHASE 4: UPDATE METRICS
                # ═══════════════════════════════════════
                layers_alive, topo_spread = get_topology_stats(nexus)
                
                # Update metrics with NaN protection
                ui.metrics["mean_confidence"] = float(torch.nan_to_num(torch.tensor(mean_conf), nan=0.0))
                ui.metrics["gradient_norm"] = float(torch.nan_to_num(torch.tensor(grad_norm), nan=0.0))
                
                ds = nexus.core.dream_seeder.dream_strength.item()
                ui.metrics["dream_strength"] = float(torch.nan_to_num(torch.tensor(ds), nan=0.0))
                
                ui.metrics["bonds"] = len(nexus.core.broker.active_bonds)
                ui.metrics["bonds_formed_total"] = nexus.core.broker.bonds_formed_total
                ui.metrics["bonds_dissolved_total"] = nexus.core.broker.bonds_dissolved_total
                
                dl = total_loss / 20.0
                ui.metrics["dream_loss"] = float(torch.nan_to_num(torch.tensor(dl), nan=0.0))
                ui.metrics["consolidation_loss"] = 0.0
                ui.metrics["layers_alive"] = layers_alive
                ui.metrics["topology_spread"] = float(torch.nan_to_num(torch.tensor(topo_spread), nan=0.0))

                live.update(ui.update_layout())

                # ═══════════════════════════════════════
                # PHASE 5: PERIODIC SAVE
                # ═══════════════════════════════════════
                if cycle % args.save_interval == 0:
                    if ckpt_path.exists():
                        backup_path = ckpt_path.with_name(f"{ckpt_path.stem}_BACKUP.pt")
                        import shutil
                        shutil.copy2(ckpt_path, backup_path)

                    torch.save({
                        'model_state_dict': nexus.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, str(ckpt_path) + '.tmp')
                    import os
                    os.replace(str(ckpt_path) + '.tmp', str(ckpt_path))
                    ui.log(f"Checkpoint saved at dream cycle {cycle} (Backup updated).")

                # Log milestones
                if cycle % 25 == 0:
                    ui.log(f"Cycle {cycle} | Loss: {total_loss / 20.0:.4f} | Conf: {mean_conf:.4f} | Bonds: {len(bonds)}")

                time.sleep(0.05)  # Small pause to prevent UI flicker

            except KeyboardInterrupt:
                ui.log("Dreaming interrupted. Saving final state...")
                if ckpt_path.exists():
                    backup_path = ckpt_path.with_name(f"{ckpt_path.stem}_BACKUP.pt")
                    import shutil
                    shutil.copy2(ckpt_path, backup_path)
                torch.save({
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, str(ckpt_path) + '.tmp')
                import os
                os.replace(str(ckpt_path) + '.tmp', str(ckpt_path))
                ui.log("Final state saved. Exiting.")
                break


if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser(description="PCEI Dreaming Chamber — Phase 3 Self-Resonance")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_micro.pt")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-interval", type=int, default=50, help="Save every N dream cycles")
    parser.add_argument("--d-model-text", type=int, default=128)
    parser.add_argument("--d-model-img", type=int, default=128)
    parser.add_argument("--d-model-audio", type=int, default=64)
    parser.add_argument("--layers-text", type=int, default=2)
    parser.add_argument("--layers-img", type=int, default=1)
    parser.add_argument("--layers-audio", type=int, default=1)
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dreaming Chamber closed safely.[/yellow]")
        sys.exit(0)
