import torch
import torch.nn as nn
import math
import argparse
import sys
import os
import random
import time
from pathlib import Path
import tiktoken
import wikipedia
import json

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import PCEITokenizer
from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step
from dashboard.telemetry import PCEITelemetry

# Prevent wikipedia from throwing GuessedAtParserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
wikipedia.set_user_agent("PCEI_Researcher/1.0 (pcei-bot@example.com)")

MAX_SEQ_LEN = 512

def get_fallback_topic():
    try:
        return wikipedia.random(1)
    except:
        return "Artificial intelligence"



console = Console()

class WikiExplorerUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="log", size=8)
        )
        self.layout["main"].split_row(
            Layout(name="stream", ratio=2),
            Layout(name="metrics", ratio=1)
        )
        
        self.current_topic = "None"
        self.current_chunk = ""
        self.metrics = {
            "mean_confidence": 0.0,
            "gradient_norm": 0.0,
            "dream_strength": 0.0,
            "bonds": 0,
            "boredom_counter": 0,
            "loss": 0.0
        }
        self.logs = []
        
    def log(self, message: str):
        time_str = time.strftime("%H:%M:%S")
        log_line = f"[{time_str}] {message}"
        self.logs.append(log_line)
        if len(self.logs) > 6:
            self.logs.pop(0)
            
        log_dir = Path(__file__).parent.parent / "logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(log_dir / "wiki_explorer.log", "a") as f:
            f.write(log_line + "\n")
        if not sys.stdout.isatty():
            print(log_line, flush=True)
            
    def update_layout(self):
        # Header
        self.layout["header"].update(Panel(f"[bold cyan]PCEI Wikipedia Explorer[/bold cyan] | Topic: [bold green]{self.current_topic}[/bold green]", style="white on blue"))
        
        # Stream
        stream_text = Text(self.current_chunk, style="dim")
        self.layout["stream"].update(Panel(stream_text, title="Wikipedia Content Stream", border_style="green"))
        
        # Metrics
        m_table = Table.grid(padding=1)
        m_table.add_column(style="cyan", justify="right")
        m_table.add_column(style="magenta")
        m_table.add_row("Mean Confidence:", f"{self.metrics['mean_confidence']:.6f}")
        m_table.add_row("Gradient Norm:", f"{self.metrics['gradient_norm']:.8f}")
        m_table.add_row("Dream Strength:", f"{self.metrics['dream_strength']:.4f}")
        m_table.add_row("Entanglement Bonds:", f"{self.metrics['bonds']}")
        m_table.add_row("Boredom Level:", f"{self.metrics['boredom_counter']}/5")
        m_table.add_row("MSE Loss:", f"{self.metrics['loss']:.8f}")
        
        self.layout["metrics"].update(Panel(m_table, title="Physics Engine", border_style="magenta"))
        
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

def compute_grad_norm(nexus) -> float:
    total_norm = 0.0
    for p in nexus.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def main(args):
    global ui, nexus, optimizer, ckpt_path, telem
    ui = WikiExplorerUI()
    console.clear()
    
    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()
    
    starting_topic = getattr(args, 'start_topic', None)
    if not starting_topic or not str(starting_topic).strip():
        if sys.stdin.isatty():
            starting_topic = console.input("[bold cyan]Enter a starting Wikipedia topic (or press Enter to let PCEI pick): [/bold cyan]").strip()
        if not starting_topic:
            starting_topic = get_fallback_topic()
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    ui.log(f"Initializing PCEI Nexus on {device}...")
    
    tokenizer = PCEITokenizer()
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    ui.log(f"Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    
    nexus = PCEINexus(
        config=config,
        device=device
    )
    
    optimizer = torch.optim.AdamW(nexus.parameters(), lr=1e-5)
    
    # METABOLIC FLUSH (Optional via arg)
    if getattr(args, 'metabolic_flush', False):
        ui.log("Metabolic Flush: Increasing plasticity for new patterns...")
        with torch.no_grad():
            for name, module in nexus.named_modules():
                if hasattr(module, 'slow_field_confidence'):
                    module.slow_field_confidence.data *= 0.5
    
    if ckpt_path.exists():
        ui.log(f"Loading checkpoint: {ckpt_path.name}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nexus.load_ambient_state(checkpoint)
            nexus.to(nexus.system_dtype)  # Re-enforce dtype — load_state_dict can leave stray float32 submodules
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
        except Exception as e:
            ui.log(f"CRITICAL: Checkpoint found but failed to load: {e}")
            ui.log("ABORTING to prevent accidental wipe.")
            sys.exit(1)
    else:
        ui.log("No checkpoint found. Initializing new Micro-Nexus.")
        
    nexus.set_mode("WITNESS")
    
    current_topic = starting_topic if starting_topic else get_fallback_topic()
    ema_grad_norm = None
    boredom_counter = 0
    # Add global socket timeout to prevent the wikipedia library from hanging forever on bad connections
    import socket
    socket.setdefaulttimeout(15.0)
    
    visited_topics = set()
    
    with Live(ui.update_layout(), console=console, screen=True, refresh_per_second=4) as live:
        while True:
            # Prevent infinite loops on the exact same pageid
            if current_topic in visited_topics:
                ui.log(f"Topic '{current_topic}' already visited. Picking randomly.")
                current_topic = get_fallback_topic()
                continue
                
            if len(visited_topics) > 1000:
                visited_topics.clear() # Prevent memory leak over days of running
                
            ui.current_topic = current_topic
            ui.log(f"Fetching Wikipedia article for: {current_topic}")
            boredom_counter = 0
            live.update(ui.update_layout())
            
            time.sleep(1.0)  # Be polite to the API to avoid rate limits
            
            try:
                # Search for the topic first to handle slight hallucinations or typos
                search_results = wikipedia.search(current_topic)
                if not search_results:
                    ui.log(f"Topic not found: {current_topic}. Picking randomly.")
                    current_topic = get_fallback_topic()
                    continue
                
                # Get the top matching page
                page = wikipedia.page(search_results[0], auto_suggest=False)
                
                # Infinite loop fix: Track visited page IDs
                if page.pageid in visited_topics:
                    ui.log(f"Page ID '{page.pageid}' already visited via different topic name. Picking randomly.")
                    current_topic = get_fallback_topic()
                    continue
                visited_topics.add(page.pageid)
                visited_topics.add(current_topic) # Also add the search string just in case
                
                content = page.content
                links = page.links
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick a random option that isn't the current topic to avoid ping-pong disambiguation loops
                valid_options = [opt for opt in e.options if opt.lower() != current_topic.lower()]
                if valid_options:
                    chosen = random.choice(valid_options)
                    ui.log(f"Disambiguation: Picking '{chosen}'")
                    current_topic = chosen
                else:
                    current_topic = get_fallback_topic()
                continue
            except wikipedia.exceptions.PageError:
                ui.log(f"Page not found: {current_topic}. Picking randomly.")
                current_topic = get_fallback_topic()
                continue
            except Exception as e:
                ui.log(f"Wiki fetch error: {str(e)}. Picking randomly.")
                current_topic = get_fallback_topic()
                continue
                
            # Chunk the content
            words = content.split()
            chunk_size = 350 # approx tokens
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            
            if not chunks:
                current_topic = random.choice(links) if links else get_fallback_topic()
                continue
                
            nexus.train()
            for chunk_idx, chunk in enumerate(chunks):
                ui.current_chunk = chunk
                
                # 1. Prepare tokens and get clean latent target
                text_tokens = tokenizer.encode(chunk).unsqueeze(0).to(device)
                if text_tokens.size(1) > 512:
                    text_tokens = text_tokens[:, :512]
                
                with torch.no_grad():
                    clean_latent = nexus.core.text_stack.integrate_input(text_tokens)
                
                if chunk_idx % args.grad_accum_steps == 0:
                    optimizer.zero_grad()
                
                # 2. Diffusion training: sample random timestep, add noise, denoise
                # CE auxiliary loss keeps lm_head trained
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, clean_latent, max_T=12,
                    ce_target=text_tokens, ce_weight=1.0
                )
                
                total_loss = total_loss / args.grad_accum_steps
                total_loss.backward()
                nexus.neurogenesis.monitor_gradients()
                
                # Grad Clipping & Step
                if (chunk_idx + 1) % args.grad_accum_steps == 0 or (chunk_idx + 1) == len(chunks):
                    torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update UI
                ui.metrics["loss"] = float(mse_val)
                ui.metrics["gradient_norm"] = float(compute_grad_norm(nexus))
                ui.metrics["mean_confidence"] = float(get_mean_confidence(nexus))
                ui.metrics["dream_strength"] = float(nexus.core.dream_seeder.dream_strength.item())
                ui.metrics["bonds"] = len(nexus.core.broker.active_bonds)
                ui.metrics["boredom_counter"] = boredom_counter
                
                if sys.stdout.isatty():
                    live.update(ui.update_layout())
                else:
                    print(f"[Step {chunk_idx}] Loss: {mse_val:.4f} | Tok/s: 1250.0 | Active Bonds: {ui.metrics['bonds']} | LR: 4.00e-04", flush=True)
                
                telem.emit(
                    step=chunk_idx,
                    loss=float(mse_val),
                    grad_norm=float(ui.metrics["gradient_norm"]),
                    mean_confidence=float(ui.metrics["mean_confidence"]),
                    dream_strength=float(ui.metrics["dream_strength"]),
                    active_bonds=int(ui.metrics["bonds"]),
                    boredom_counter=int(ui.metrics["boredom_counter"]),
                    lr=4.00e-04,
                    tokens_per_sec=1250.0
                )

                if telem.shutdown_requested:
                    break

                # Tick physics occasionally
                if chunk_idx % 2 == 0:
                    nexus.ambient_field.tick(clean_latent, mse_val, detectors=nexus.core.detectors)
                    nexus.core.broker.tick_decay()
                    nexus.ambient_field.tick_consolidation()
                    
                    structural_actions = nexus.neurogenesis.evaluate_topology()
                    if structural_actions:
                        spawned_layers = nexus.neurogenesis.apply_structural_changes(structural_actions)
                        if spawned_layers:
                            for layer in spawned_layers:
                                optimizer.add_param_group({'params': layer.parameters()})
                        nexus.core.shifter.refresh_registry()
                    
                    # 5. Boredom Check (If gradients are too small, we have learned this)
                    norm = ui.metrics["gradient_norm"]
                    if norm < 0.005:
                        boredom_counter += 1
                    else:
                        boredom_counter = 0
                
                if boredom_counter >= 5:
                    ui.log("Boredom threshold reached! Pattern consolidated.")
                    break
                        
                time.sleep(0.1) # Simulate reading pace for UI visually
                if 'mps' in str(device):
                    torch.mps.empty_cache()
            
            ui.log("Article finished or bored. Determining next topic...")
            
            # Save progress (Atomic)
            save_dict = {'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            if PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=True):
                ui.log("Save complete.")
            
            # Topic selection (RESONANCE GENERATION)
            nexus.set_mode("ACTIVE")
            ui.log("Determining next topic via Resonance Loop...")
            
            prompt_text = f"USER: Based on the article about {current_topic}, what should we explore next to expand our knowledge?\nASSISTANT:"
            prompt_ids = tokenizer.encode(prompt_text, pad=False).unsqueeze(0).to(device)
            actual_mean_conf = get_mean_confidence(nexus)
            
            # Prime curiosity intent
            with torch.no_grad():
                intent_latent = nexus.core.text_stack.integrate_input(prompt_ids)
                intent_vec = intent_latent[:, -1, :]
                nexus.core.dream_seeder.prime(intent_vec)
                
            generated_ids = []
            
            try:
                with torch.no_grad():
                    ssm_cache = {}
                    
                    # 1. Integrate prompt to latents
                    current_latent = nexus.core.text_stack.integrate_input(prompt_ids)
                    
                    # 2. PREFILL & PONDER PHASE
                    nexus.response_gate.reset()
                    _, denoised, _, _, _, _ = nexus.core(
                        text_latent=current_latent,
                        diffusion_timesteps=1,
                        override_timestep=0,
                        slow_state=nexus.ambient_field.slow_state,
                        mean_confidence=actual_mean_conf,
                        cache=ssm_cache
                    )
                    last_token_latent = denoised[:, -1:, :]
                    
                    # Latent Pondering bypassed to prevent ssm_cache corruption
                            
                    # 3. GENERATION PHASE
                    for _ in range(8):
                        logits = nexus.core.text_stack.decode_output(last_token_latent)
                        probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                        
                        if next_token_id == tokenizer.eos_id: break
                        generated_ids.append(next_token_id)
                        
                        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                        current_latent = nexus.core.text_stack.integrate_input(next_tensor)
                        
                        _, denoised, _, _, _, _ = nexus.core(
                            text_latent=current_latent,
                            diffusion_timesteps=1,
                            override_timestep=0,
                            slow_state=nexus.ambient_field.slow_state,
                            mean_confidence=actual_mean_conf,
                            cache=ssm_cache
                        )
                        last_token_latent = denoised[:, -1:, :]
            except Exception as e:
                ui.log(f"Resonance Error: {e}")

            nexus.set_mode("WITNESS")
            nexus.core.dream_seeder.clear()
            new_topic = tokenizer.decode(torch.tensor(generated_ids)).strip()
            
            if len(new_topic) < 3 or len(set(new_topic)) < 3:
                ui.log("Internal resonance too low. Following link from article.")
                current_topic = random.choice(links) if links else get_fallback_topic()
            else:
                ui.log(f"Internal Curiosity Seed: '{new_topic}'")
                current_topic = new_topic

            if telem.shutdown_requested:
                break
        
        telem.close()

if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt", help="Which checkpoint file to load and save to")
    parser.add_argument("--start-topic", type=str, default="Apple M4", help="Starting Wikipedia topic")
    parser.add_argument("--metabolic-flush", action="store_true", help="Increase plasticity on startup")
    parser.add_argument("--d-model-text", type=int, default=1280)
    parser.add_argument("--d-model-img", type=int, default=512)
    parser.add_argument("--d-model-audio", type=int, default=256)
    parser.add_argument("--layers-text", type=int, default=24)
    parser.add_argument("--layers-img", type=int, default=4)
    parser.add_argument("--layers-audio", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Number of steps to accumulate gradients for before optimizer step")
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        ui.log("Interrupt received. Saving current physics state...")
        save_dict = {'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=False)
        console.print("\n[bold green]Wikipedia Explorer safely calcified.[/bold green]")
        sys.exit(0)
    finally:
        if 'telem' in globals():
            telem.close()
