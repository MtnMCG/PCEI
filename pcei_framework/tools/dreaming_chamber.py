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

from src.tokenizer import PCEITokenizer
from dashboard.telemetry import PCEITelemetry


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
        print(log_line, flush=True)

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
    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()

    ui = DreamingUI()
    if sys.stdout.isatty():
        console.clear()

    # Prompt for optional seed topic if interactive, or use --seed CLI argument
    seed_input = getattr(args, "seed", None) or ""
    if not seed_input and sys.stdin.isatty():
        try:
            seed_input = console.input(
                "[bold cyan]Enter a seed topic to prime the dreams (or press Enter for pure self-resonance): [/bold cyan]"
            )
        except (EOFError, KeyboardInterrupt):
            seed_input = ""
    seed_input = seed_input.strip()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = PCEITokenizer()

    from src.config import PCEIConfig
    config = PCEIConfig.load()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    save_ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.output_name if getattr(args, "output_name", None) else ckpt_path
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio
    ui.log(f"Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")

    nexus = PCEINexus(
        config=config,
        device=device
    )

    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)
    if ckpt_path.exists():
        ui.log(f"Loading checkpoint: {ckpt_path.name} (Output will save to {save_ckpt_path.name})")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nexus.load_ambient_state(checkpoint)
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
                    ui.log(f"Sanitizer: Cleaned {nan_count} corrupted weights.")
                if revived_count > 0:
                    ui.log(f"Defibrillator: Revived {revived_count} collapsed layers.")
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Force all optimizer states to bfloat16 for MPS compatibility
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(nexus.system_dtype)
                except:
                    pass
            ui.log("Foundation loaded successfully.")
            if getattr(args, 'metabolic_flush', False):
                ui.log("Metabolic Flush: Resetting slow-field confidence...")
                with torch.no_grad():
                    for m in nexus.core.modules():
                        if isinstance(m, PlasticLayer):
                            m.slow_field_confidence.zero_()
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
    ui.log("Commencing self-resonance dream cycles...")
    is_tty = sys.stdout.isatty()

    def run_dream_iteration(cycle: int, live_handle=None):
        nonlocal optimizer
        ui.dream_cycle = cycle
        mean_conf = get_mean_confidence(nexus)

        # ═══════════════════════════════════════
        # PHASE 1 & 2: AUTOREGRESSIVE SELF-PLAY & DECODE
        # ═══════════════════════════════════════
        with torch.no_grad():
            seq_len = random.randint(8, 16)
            
            # Start with a random common token or the user seed
            if seed_latent is not None and cycle < 50:
                input_ids = seed_tokens.clone()
            else:
                input_ids = torch.randint(1, 1000, (1, 1), device=device, dtype=torch.long)
                
            ui.dream_history.append("")
            if len(ui.dream_history) > 50:
                ui.dream_history.pop(0)
                
            # Generate thought
            for _ in range(seq_len):
                logits, t_lat, _, _, _, _ = nexus.core(
                    text_input=input_ids,
                    diffusion_timesteps=1,
                    override_timestep=0,
                    slow_state=nexus.ambient_field.slow_state,
                    mean_confidence=mean_conf
                )
                next_token_logits = logits[:, -1, :] / 0.9 # Temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                current_text = tokenizer.decode(input_ids[0])
                ui.dream_history[-1] = current_text.strip()[:120]
                if live_handle is not None:
                    live_handle.update(ui.update_layout())
                
            dream_tokens = input_ids
            # Get final latent representation for Ambient Field
            _, dream_output, _, _, _, _ = nexus.core(
                text_input=dream_tokens,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=mean_conf
            )
            dream_text = tokenizer.decode(dream_tokens[0])
            ui.dream_history[-1] = dream_text.strip()[:120]
            
        # Evaluate Confidence
        conf_score = get_mean_confidence(nexus)
        
        # Training Phase (Reinforcement)
        # Reinforce when confidence is >= 0.25 (750M foundation runs at ~0.4601)
        if conf_score >= 0.25:
            optimizer.zero_grad()
            
            # Dummy clean_latent just to satisfy function signature (ignored by text modality)
            dummy_latent = torch.zeros(1, dream_tokens.size(1), d_text, device=device, dtype=nexus.system_dtype)
            
            total_loss_val, mse_val, _, _ = diffusion_training_step(
                nexus, clean_latent=dummy_latent, ce_target=dream_tokens, ce_weight=1.0, modality='text'
            )
            
            dist_loss = nexus.core.shifter.calculate_metabolic_loss() if hasattr(nexus.core.shifter, 'calculate_metabolic_loss') else 0
            total_loss_val = total_loss_val + (dist_loss * 0.01)
            
            total_loss_val.backward()
            nexus.neurogenesis.monitor_gradients()
            total_loss = total_loss_val.item()
            
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()
        else:
            total_loss = 0.0
            mse_val = 0.0
            total_loss_val = None
            
        grad_norm = compute_grad_norm(nexus)

        # ═══════════════════════════════════════
        # PHASE 3: PHYSICS TICK
        # ═══════════════════════════════════════
        nexus.ambient_field.tick(dream_output, total_loss / 20.0, detectors=nexus.core.detectors)
        nexus.adjust_plasticity(optimizer)

        bonds = nexus.core.broker.measure_entanglement(nexus.core.detectors)
        nexus.core.broker.tick_decay()
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)

        # Neurogenesis check (every 10 cycles)
        if cycle % 10 == 0:
            structural_actions = nexus.neurogenesis.evaluate_topology()
            if structural_actions:
                spawned_layers = nexus.neurogenesis.apply_structural_changes(structural_actions)
                if spawned_layers:
                    optimizer = torch.optim.AdamW(nexus.parameters(), lr=args.lr, weight_decay=0.01)
                nexus.core.shifter.refresh_registry()
                ui.log(f"Neurogenesis event: {len(structural_actions)} structural changes. Optimizer rebuilt.")

        # ═══════════════════════════════════════
        # PHASE 4: UPDATE METRICS & LOG OUTPUT
        # ═══════════════════════════════════════
        layers_alive, topo_spread = get_topology_stats(nexus)
        
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

        if live_handle is not None:
            live_handle.update(ui.update_layout())
        else:
            clean_dream = dream_text.strip().replace('\n', ' ')
            reinforce_str = f"Reinforced (Loss: {ui.metrics['dream_loss']:.4f})" if conf_score > 0.5 else "Passive (Conf <= 0.5)"
            print(f"\n--- Dream Cycle {cycle} [Seed: {ui.seed_topic}] ---", flush=True)
            print(f"DREAM: \"{clean_dream}\"", flush=True)
            print(f"[Cycle {cycle}] Status: {reinforce_str} | Conf: {ui.metrics['mean_confidence']:.4f} | Bonds: {ui.metrics['bonds']} | LR: {args.lr:.2e}", flush=True)

        telem.emit(
            step=cycle,
            loss=ui.metrics["dream_loss"],
            mean_confidence=ui.metrics["mean_confidence"],
            grad_norm=ui.metrics["gradient_norm"],
            dream_strength=ui.metrics["dream_strength"],
            active_bonds=ui.metrics["bonds"],
            bonds_formed_total=ui.metrics["bonds_formed_total"],
            bonds_dissolved_total=ui.metrics["bonds_dissolved_total"],
            consolidation_loss=ui.metrics["consolidation_loss"],
            layers_alive=ui.metrics["layers_alive"],
            topology_spread=ui.metrics["topology_spread"],
            lr=args.lr,
            dream_text=dream_text.strip()[:150],
            seed_topic=ui.seed_topic
        )

        # ═══════════════════════════════════════
        # PHASE 5: PERIODIC SAVE & GC
        # ═══════════════════════════════════════
        if cycle % args.save_interval == 0:
            save_dict = {
                'model_state_dict': nexus.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ambient_slow_state': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
                'ambient_fast_state': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.fast_state,
                'ambient_layer_emas': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.layer_emas,
                'active_bonds': getattr(nexus.core, 'broker', None) and nexus.core.broker.active_bonds,
            }
            if PCEINexus.save_safe_checkpoint(nexus, save_dict, save_ckpt_path, create_backup=True):
                ui.log(f"Checkpoint saved at dream cycle {cycle} to {save_ckpt_path.name}.")

        if cycle % 25 == 0:
            ui.log(f"Cycle {cycle} | Loss: {total_loss / 20.0:.4f} | Conf: {mean_conf:.4f} | Bonds: {len(bonds)}")

        if 'total_loss_val' in locals() and total_loss_val is not None:
            del total_loss_val
        if cycle % 50 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        time.sleep(0.05)

    try:
        if is_tty:
            with Live(ui.update_layout(), console=console, screen=True, refresh_per_second=2) as live:
                cycle = 0
                while True:
                    if telem.shutdown_requested or (args.max_cycles and cycle >= args.max_cycles):
                        break
                    cycle += 1
                    run_dream_iteration(cycle, live_handle=live)
        else:
            cycle = 0
            while True:
                if telem.shutdown_requested or (args.max_cycles and cycle >= args.max_cycles):
                    break
                cycle += 1
                run_dream_iteration(cycle, live_handle=None)
    except KeyboardInterrupt:
        ui.log("Dreaming interrupted by user.")

    # Save final consolidated state
    ui.log(f"Saving final state to {save_ckpt_path.name}...")
    save_dict = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ambient_slow_state': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
        'ambient_fast_state': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.fast_state,
        'ambient_layer_emas': getattr(nexus, 'ambient_field', None) and nexus.ambient_field.layer_emas,
        'active_bonds': getattr(nexus.core, 'broker', None) and nexus.core.broker.active_bonds,
    }
    PCEINexus.save_safe_checkpoint(nexus, save_dict, save_ckpt_path, create_backup=False)
    ui.log(f"Final state successfully saved to {save_ckpt_path.name}. Exiting.")

    telem.close()


if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser(description="PCEI Dreaming Chamber — Phase 3 Self-Resonance")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--output-name", type=str, default=None, help="Save final checkpoint to this name")
    parser.add_argument("--max-cycles", "--cycles", type=int, default=None, help="Exit after N dream cycles")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-interval", type=int, default=50, help="Save every N dream cycles")
    parser.add_argument("--d-model-text", type=int, default=1280)
    parser.add_argument("--d-model-img", type=int, default=512)
    parser.add_argument("--d-model-audio", type=int, default=256)
    parser.add_argument("--layers-text", type=int, default=24)
    parser.add_argument("--layers-img", type=int, default=4)
    parser.add_argument("--layers-audio", type=int, default=4)
    parser.add_argument("--seed", type=str, default="", help="Optional seed topic to prime dreams")
    parser.add_argument("--metabolic-flush", action="store_true", default=False, help="Reset slow-field confidence to zero before training")
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dreaming Chamber closed safely.[/yellow]")
        sys.exit(0)
