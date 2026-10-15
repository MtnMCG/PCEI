import os
import torch
import torch.nn as nn
import argparse
import sys
import json
import random
import time
import signal
import gc
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step
from scripts.train import PCEITokenizer
from dashboard.telemetry import PCEITelemetry

console = Console()
MAX_SEQ_LEN = 256

def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            val = module.slow_field_confidence.item()
            if torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                val = 0.0
            conf_vals.append(val)
    return sum(conf_vals) / max(len(conf_vals), 1)


class StreamedCategoryLoader:
    """
    O(1) RAM streaming loader.
    Never loads massive text lists into memory. Maps files and byte-seeks randomly.
    """
    def __init__(self, category_id: int):
        self.category_id = category_id
        self.txt_files = []
        self.txt_sizes = []
        self.jsonl_files = []
        self.jsonl_sizes = []
        self.jsonl_indices = {}
        
        self._map_files()
        
    def _map_files(self):
        data_dir = Path(__file__).parent.parent / "data"
        if self.category_id == 1:
            target_dir = data_dir / "raw" / "text" / "gutenberg"
            if target_dir.exists():
                self.txt_files = list(target_dir.glob("*.txt"))
        elif self.category_id == 2:
            target_dir = data_dir / "raw" / "text" / "modern_books"
            if target_dir.exists():
                self.txt_files = list(target_dir.glob("*.txt"))
        elif self.category_id == 3:
            wiki_dir = data_dir / "raw" / "text"
            if wiki_dir.exists():
                self.txt_files.extend(list(wiki_dir.glob("wiki_*.txt")))
            mass_dir = data_dir / "raw" / "text" / "mass_reasoning"
            if mass_dir.exists():
                self.txt_files.extend(list(mass_dir.glob("*.txt")))
            hq_dir = data_dir / "high_quality"
            if hq_dir.exists():
                self.jsonl_files.extend(list(hq_dir.glob("*.jsonl")))
            sync_dir = data_dir / "synthetic"
            if sync_dir.exists():
                self.jsonl_files.extend(list(sync_dir.glob("*.jsonl")))
        elif self.category_id == 4:
            target_dir = data_dir / "raw" / "text" / "mass_reasoning"
            if target_dir.exists():
                self.txt_files = list(target_dir.glob("*.txt"))
                
        # Compute sizes for proportional sampling
        self.txt_sizes = [f.stat().st_size for f in self.txt_files]
        self.jsonl_sizes = [f.stat().st_size for f in self.jsonl_files]
                
        for fpath in self.jsonl_files:
            self._index_jsonl(fpath)
            
    def _index_jsonl(self, fpath):
        offsets = []
        with open(fpath, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        self.jsonl_indices[fpath] = offsets

    def get_random_chunk(self):
        total_txt_bytes = sum(self.txt_sizes)
        total_jsonl_bytes = sum(self.jsonl_sizes)
        
        if total_txt_bytes == 0 and total_jsonl_bytes == 0:
            return "blank placeholder"
            
        choice = random.choices(['txt', 'jsonl'], weights=[total_txt_bytes, total_jsonl_bytes])[0]
        
        if choice == 'txt':
            fpath = random.choices(self.txt_files, weights=self.txt_sizes)[0]
            size = fpath.stat().st_size
            if size < 5000:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                seek_pos = random.randint(0, size - 5000)
                with open(fpath, 'rb') as f:
                    f.seek(seek_pos)
                    f.readline()
                    chunk = f.read(5000).decode('utf-8', errors='ignore')
                    return chunk
        else:
            fpath = random.choices(self.jsonl_files, weights=self.jsonl_sizes)[0]
            offsets = self.jsonl_indices[fpath]
            if not offsets: return "blank placeholder"
            
            seek_pos = random.choice(offsets)
            with open(fpath, 'rb') as f:
                f.seek(seek_pos)
                line = f.readline().decode('utf-8', errors='ignore')
                try:
                    record = json.loads(line)
                    if 'prompt' in record and 'response' in record:
                        p = record['prompt']
                        r = record['response']
                        if p.startswith("USER:"):
                            return f"{p}{r}"
                        else:
                            return f"USER: {p}\nASSISTANT: {r}"
                    elif 'text' in record:
                        return record['text']
                    elif 'instruction' in record:
                        return record.get('instruction', '') + "\n" + record.get('output', '')
                except:
                    return "blank placeholder"
        return "blank placeholder"

def get_text_batch(loader, tokenizer, batch_size, device):
    batch_tokens = []
    for _ in range(batch_size):
        text = loader.get_random_chunk()
        safe_text = text[:5000] if isinstance(text, str) else "blank placeholder"
        if not safe_text.strip(): safe_text = "blank placeholder"
        batch_tokens.append(tokenizer.encode(safe_text, max_len=MAX_SEQ_LEN))
    
    return torch.stack(batch_tokens).to(device)


import threading
import queue

class PrefetchLoader:
    """
    Wraps a StreamedCategoryLoader with a background thread to prefetch and tokenize batches.
    Prevents GPU starvation (9% utilization) by ensuring batches are always ready in memory.
    Bounded maxsize (e.g., 50 batches) strictly prevents RAM explosion.
    """
    def __init__(self, loader, tokenizer, batch_size, device, maxsize=50):
        self.loader = loader
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.q = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # We process on CPU, only pushing to MPS/CUDA in the main thread to avoid cross-thread context issues
        cpu_device = torch.device("cpu")
        while not self.stop_event.is_set():
            batch = get_text_batch(self.loader, self.tokenizer, self.batch_size, cpu_device)
            # This blocks if queue is full (backpressure prevents RAM explosion)
            self.q.put(batch)

    def get_batch(self):
        # Fetches batch from queue and transfers to compute device
        batch = self.q.get()
        return batch.to(self.device)

    def stop(self):
        self.stop_event.set()
        # Drain queue to unblock the worker thread so it can exit
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Number of steps to accumulate gradients for before optimizer step")
    parser.add_argument("--curriculum", type=str, default="3-1000", help="Curriculum string e.g. 1-300,2-10,3-50")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]PCEI Reading Room[/bold cyan] initialized on {device}")
    console.print("[bold yellow]RAM Optimization ACTIVE (O(1) Memory Footprint)[/bold yellow]")
    
    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()
    
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
    
    nexus = PCEINexus(
        config=config,
        device=device
    )
    
    if ckpt_path.exists():
        console.print(f"[green]Loading checkpoint {ckpt_path.name}...[/green]")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
        nexus.load_ambient_state(checkpoint)
        nexus.to(device)
        del checkpoint
        gc.collect() # Immediate cleanup
    else:
        console.print(f"[yellow]Warning: Checkpoint {ckpt_path.name} not found. Starting from scratch.[/yellow]")
        
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)
    tokenizer = PCEITokenizer(vocab_size=config.vocab_size)
    
    console.print("\n[bold]Available Categories:[/bold]")
    console.print("  [1] Philosophy (Gutenberg)")
    console.print("  [2] Modern Books")
    console.print("  [3] The Ultimate Mix (Philosophy + Wiki + TinyStories + Cosmopedia + SFT Conversational Data)")
    console.print("  [4] Mass Reasoning Text only (TinyStories, Cosmopedia, etc.)")
    
    curriculum_str = args.curriculum
    if not curriculum_str:
        console.print("[red]No curriculum provided. Exiting.[/red]")
        sys.exit(0)
        
    tasks = []
    for part in curriculum_str.split(','):
        if '-' in part:
            cat_id, steps = part.split('-')
            tasks.append((int(cat_id.strip()), int(steps.strip())))
            
    nexus.set_mode("WITNESS")
    
    def safe_save(step=None, tag=""):
        save_dict = {
            'model_state_dict': nexus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': step if step is not None else 0
        }
        PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=True)
        console.print(f"\n[bold green]Atomic Checkpoint Saved! {tag}[/bold green]")
        del save_dict
        gc.collect()

    loaded_categories = {}
    
    try:
        for cat_id, steps in tasks:
            console.print(f"\n[bold magenta]Loading Category {cat_id}...[/bold magenta]")
            if cat_id not in loaded_categories:
                loaded_categories[cat_id] = StreamedCategoryLoader(cat_id)
                
            base_loader = loaded_categories[cat_id]
            file_count = len(base_loader.txt_files) + len(base_loader.jsonl_files)
            if file_count == 0:
                console.print(f"[bold red]Category {cat_id} has no mapped files! Skipping.[/bold red]")
                continue
                
            console.print(f"[green]Mapped {file_count} files for Category {cat_id} (Zero RAM footprint). Starting {steps} steps.[/green]")
            
            # Start background prefetcher to saturate the GPU
            prefetcher = PrefetchLoader(base_loader, tokenizer, args.batch_size, device, maxsize=50)
            
            import collections
            recent_rows = collections.deque(maxlen=15)
            
            table = Table(title=f"Curriculum: Category {cat_id} for {steps} steps")
            table.add_column("Step", justify="right", style="cyan")
            table.add_column("Loss", justify="right", style="magenta")
            table.add_column("Mean Conf", justify="right", style="green")
            table.add_column("Speed", justify="right", style="yellow")
            
            with Live(table, refresh_per_second=4) as live:
                start_time = time.time()
                optimizer.zero_grad()
                for step in range(1, steps + 1):
                    if telem.shutdown_requested:
                        break
                        
                    tokens = prefetcher.get_batch()
                    
                    with torch.no_grad():
                        clean_latent = nexus.core.text_stack.integrate_input(tokens)
                    
                    total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                        nexus, clean_latent, max_T=12,
                        ce_target=tokens, ce_weight=1.0
                    )
                    
                    dist_loss = nexus.core.shifter.calculate_metabolic_loss() if hasattr(nexus.core.shifter, 'calculate_metabolic_loss') else 0
                    total_loss = total_loss + (dist_loss * 0.01)
                    
                    # Gradient Accumulation Scaling
                    loss_for_backward = total_loss / args.grad_accum_steps
                    loss_for_backward.backward()
                    
                    nexus.neurogenesis.monitor_gradients()
                    
                    # SCORCH EARTH GRADIENT SANITIZER
                    for p in nexus.parameters():
                            pass # nan_to_num band-aid removed
                            
                    if step % args.grad_accum_steps == 0 or step == steps:
                        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # --- CORE RESTORATION: AMBIENT FIELD ---
                        nexus.ambient_field.tick_consolidation()
                        nexus.adjust_plasticity(optimizer)
                        
                        optimizer.zero_grad()
                    
                    # --- CORE RESTORATION: NEUROGENESIS ---
                    if step % 100 == 0:
                        structural_actions = nexus.neurogenesis.evaluate_topology()
                        if structural_actions:
                            structure_changed = nexus.neurogenesis.apply_structural_changes(structural_actions)
                            if structure_changed:
                                console.print("\n[bold yellow]Neurogenesis: Topology Shifted! Rebuilding Optimizer...[/bold yellow]")
                                optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)

                    elapsed = time.time() - start_time
                    tps = (step * args.batch_size * MAX_SEQ_LEN) / elapsed if elapsed > 0 else 0
                    mean_conf = get_mean_confidence(nexus)
                    
                    if step % 5 == 0 or step == steps:
                        recent_rows.append((
                            f"{step}/{steps}",
                            f"{mse_val:.4f}",
                            f"{mean_conf:.3f}",
                            f"{tps:.0f} T/s"
                        ))
                        # Rebuild table to prevent infinite vertical growth
                        new_table = Table(title=f"Curriculum: Category {cat_id} for {steps} steps")
                        new_table.add_column("Step", justify="right", style="cyan")
                        new_table.add_column("Loss", justify="right", style="magenta")
                        new_table.add_column("Mean Conf", justify="right", style="green")
                        new_table.add_column("Speed", justify="right", style="yellow")
                        for r in recent_rows:
                            new_table.add_row(*r)
                        live.update(new_table)
                        
                    telem.emit(
                        step=step,
                        loss=mse_val,
                        lr=args.lr,
                        mean_confidence=mean_conf,
                        tokens_per_sec=tps
                    )
                        
                    # Memory Leak Prevention: Drop massive graph tensors immediately
                    del tokens, clean_latent, total_loss, dist_loss
                    
                    if step % 50 == 0:
                        if device == "mps":
                            torch.mps.empty_cache()
                        elif device == "cuda":
                            torch.cuda.empty_cache()
                            
                    if step % 100 == 0:
                        safe_save(step=step, tag=f"(Step {step})")
            
            # Stop the prefetcher thread to clean up before loading next category
            prefetcher.stop()
            del prefetcher
            
            # Explicit garbage collection after each curriculum task
            gc.collect()
    
    except KeyboardInterrupt:
        console.print("\n[bold red]Ctrl+C detected! Safely saving model state...[/bold red]")
        safe_save(tag="(Interrupt)")
        sys.exit(0)

    safe_save(tag="(Curriculum Complete)")
    console.print("\n[bold cyan]Curriculum Fully Completed![/bold cyan]")
    telem.close()

if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    main()
