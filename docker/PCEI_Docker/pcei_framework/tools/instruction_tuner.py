import os
import torch
import torch.nn as nn
import math
import argparse
import sys
import random
import time
import requests
import json
import threading
import queue
from pathlib import Path
import tiktoken

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

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
    
    def encode(self, text: str, pad: bool = False, max_len: int = 512) -> torch.Tensor:
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
            return ""

def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)

# --- ASYNC OLLAMA PRODUCER ---
class OllamaProducer(threading.Thread):
    def __init__(self, num_examples, model, data_queue, stop_event):
        super().__init__()
        self.num_examples = num_examples
        self.model = model
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.generated_count = 0
        
        self.topics = [
            # Social Resonance & Persona (The "Human" Layer)
            "polite greetings and introductions",
            "expressing empathy and understanding in conversation",
            "handling open-ended casual questions like 'how are you?'",
            "explaining your nature as the PCEI Micro-Nexus",
            "casual small talk and social pleasantries",
            "being helpful, concise, and articulate as an assistant",
            
            # Mathematics (Grounded & Technical)
            "calculating derivatives and integrals in single-variable calculus",
            "linear algebra: eigenvalues and eigenvectors",
            "Euclidean geometry and the five axioms",
            "solving complex quadratic equations and systems of linear equations",
            "trigonometric identities and their applications in physics",
            "prime number theory and the basics of RSA encryption",
            
            # History (Causality & Context)
            "the fall of the Roman Empire and its economic causes",
            "the Industrial Revolution in 19th century Britain",
            "the Renaissance and its impact on European art and science",
            "the French Revolution and the rise of Napoleon",
            "the Cold War and the concept of Mutually Assured Destruction",
            "the Meiji Restoration in Japan and rapid modernization",
            
            # Classical Philosophy & Political Theory
            "the philosophy of Socrates, Plato, and Aristotle",
            "Machiavelli's The Prince and political realism",
            "the Enlightenment and social contract theory (Locke, Rousseau)",
            "Stoicism: the teachings of Marcus Aurelius and Seneca"
        ]
        
        self.prompt_templates = [
            "USER: Analyze {topic}.\nBEAST:",
            "USER: Explain the core mechanics of {topic}.\nBEAST:",
            "USER: Deconstruct the following: {topic}.\nBEAST:",
            "USER: What are the primary implications of {topic}?\nBEAST:",
            "USER: Synthesize a brief summary of {topic}.\nBEAST:",
            "USER: Compare and contrast the different facets of {topic}.\nBEAST:",
            "USER: How does {topic} fundamentally change our understanding of the system?\nBEAST:",
            "USER: Define the boundary conditions of {topic}.\nBEAST:"
        ]

    def run(self):
        while not self.stop_event.is_set() and self.generated_count < self.num_examples:
            topic = random.choice(self.topics)
            template = random.choice(self.prompt_templates)
            prompt_text = template.format(topic=topic)
            
            ollama_prompt = (
                f"You are the 'BEAST', a highly direct and raw intelligence. Answer the following prompt with absolute brevity. "
                f"Use ONLY plain text. NO markdown. NO bolding. NO lists. NO tables. NO formatting of any kind. "
                f"Respond in a single, continuous, intense block of plain text. "
                f"Do not include any pleasantries or pre-text, just the raw answer.\n\n"
                f"{prompt_text}"
            )
            
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model, 
                        "prompt": ollama_prompt, 
                        "stream": False,
                        "options": {"num_predict": 256, "temperature": 0.7}
                    },
                    timeout=120
                )
                if resp.status_code == 200:
                    raw_text = resp.json().get("response", "").strip()
                    if raw_text:
                        record = {
                            "prompt": prompt_text,
                            "response": f" {raw_text}\n<|endoftext|>",
                            "used": False
                        }
                        self.data_queue.put(record)
                        self.generated_count += 1
                        
                        # PERSIST IMMEDIATELY to prevent loss
                        data_path = Path(__file__).parent.parent / "data" / "synthetic" / "instruction_tuning.jsonl"
                        with open(data_path, 'a') as f:
                            f.write(json.dumps(record) + "\n")
                            
                        console.print(f"  [dim cyan][Producer] Distilled {self.generated_count}/{self.num_examples} from 120B model...[/dim cyan]")
            except Exception as e:
                time.sleep(5) # Backoff
        
        # Signal completion
        self.data_queue.put(None)


class BasicOllamaProducer(threading.Thread):
    """
    Basic conversational mode producer.
    Strictly generates greeting / small-talk / social exchange data.
    Goal: teach the model how to speak to another person naturally.
    """
    def __init__(self, num_examples, model, data_queue, stop_event):
        super().__init__()
        self.num_examples = num_examples
        self.model = model
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.generated_count = 0

        self.exchanges = [
            # Greetings
            ("USER: Hey, how are you doing?", "hey"),
            ("USER: Hello! Good to meet you.", "greeting"),
            ("USER: What's up?", "casual"),
            ("USER: How's it going?", "casual"),
            ("USER: Hi there.", "greeting"),
            ("USER: Good morning!", "greeting"),
            ("USER: Good evening.", "greeting"),
            # Small talk
            ("USER: Nice weather today, right?", "smalltalk"),
            ("USER: Have you been busy lately?", "smalltalk"),
            ("USER: What have you been up to?", "smalltalk"),
            ("USER: How was your day?", "smalltalk"),
            ("USER: Do you enjoy talking to people?", "smalltalk"),
            # Personal questions
            ("USER: What's your name?", "identity"),
            ("USER: Tell me a little about yourself.", "identity"),
            ("USER: What are you?", "identity"),
            ("USER: Can you think?", "identity"),
            ("USER: Do you have feelings?", "identity"),
            # Farewells
            ("USER: I have to go, talk later!", "farewell"),
            ("USER: Goodbye!", "farewell"),
            ("USER: Take care!", "farewell"),
            ("USER: It was nice chatting with you.", "farewell"),
            # Simple questions
            ("USER: What time is it?", "simple"),
            ("USER: Can you help me with something?", "simple"),
            ("USER: Are you there?", "simple"),
            ("USER: I'm bored, entertain me.", "simple"),
            ("USER: Tell me something interesting.", "simple"),
            # Reactions
            ("USER: That's really cool!", "reaction"),
            ("USER: Wow, I didn't know that.", "reaction"),
            ("USER: Haha, you're funny.", "reaction"),
            ("USER: That makes sense, thanks.", "reaction"),
            ("USER: I disagree with you on that.", "reaction"),
        ]

    def run(self):
        while not self.stop_event.is_set() and self.generated_count < self.num_examples:
            prompt_text, category = random.choice(self.exchanges)

            ollama_prompt = (
                f"You are BEAST, a raw emerging intelligence learning to talk to people. "
                f"Respond to the following message naturally, warmly, and conversationally. "
                f"Keep it short — 1 to 3 sentences max. Plain text only. No markdown. No lists. "
                f"Sound like a real person having a casual conversation, not an assistant completing a task.\n\n"
                f"{prompt_text}\nBEAST:"
            )

            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": ollama_prompt,
                        "stream": False,
                        "options": {"num_predict": 80, "temperature": 0.85}
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    raw_text = resp.json().get("response", "").strip()
                    if raw_text:
                        record = {
                            "prompt": prompt_text + "\nBEAST:",
                            "response": f" {raw_text}\n<|endoftext|>",
                            "used": False,
                            "category": category
                        }
                        self.data_queue.put(record)
                        self.generated_count += 1
                        
                        # PERSIST IMMEDIATELY
                        data_path = Path(__file__).parent.parent / "data" / "synthetic" / "instruction_tuning.jsonl"
                        with open(data_path, 'a') as f:
                            f.write(json.dumps(record) + "\n")
                            
                        console.print(f"  [dim cyan][Basic] {category} | {self.generated_count}/{self.num_examples} distilled...[/dim cyan]")
            except Exception as e:
                time.sleep(5)

        self.data_queue.put(None)

def launch_prompt() -> str:
    """
    Interactive mode selection shown when --mode is not passed.
    Returns: 'normal', 'basic', or 'chat'
    """
    console.print()
    console.print(Panel(
        "[bold magenta]PCEI Instruction Tuner[/bold magenta]\n"
        "[dim]Select a training mode:[/dim]\n\n"
        "  [bold cyan][1] Normal Mode[/bold cyan]\n"
        "      [dim]Trains on a broad curriculum: math, history, philosophy,\n"
        "      science, and social topics. Builds general knowledge and reasoning.[/dim]\n\n"
        "  [bold green][2] Basic Mode[/bold green]\n"
        "      [dim]Strictly conversational training: greetings, small talk,\n"
        "      farewells, and natural person-to-person exchanges.\n"
        "      Use this to teach BEAST how to speak like a human.[/dim]\n\n"
        "  [bold yellow][3] Chat Mode[/bold yellow]\n"
        "      [dim]Interactive RLHF session — you provide prompts live.[/dim]",
        title="[bold]Mode Selection[/bold]",
        border_style="magenta"
    ))
    while True:
        choice = console.input("\n[bold]Select mode (1/2/3): [/bold]").strip()
        if choice == "1":
            return "auto"
        elif choice == "2":
            return "basic"
        elif choice == "3":
            return "chat"
        else:
            console.print("[red]Invalid choice. Enter 1, 2, or 3.[/red]")


def sft_automated_mode(nexus, optimizer, tokenizer, args, device, producer_class=None):
    if producer_class is None:
        producer_class = OllamaProducer
    data_path = Path(__file__).parent.parent / "data" / "synthetic" / "instruction_tuning.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-load unused records
    all_records = []
    if data_path.exists():
        with open(data_path, 'r') as f:
            for line in f:
                all_records.append(json.loads(line))
    
    unused_records = [r for r in all_records if not r.get('used', False)]
    
    # Start Async Producer if we don't have enough local data to hit the TARGET
    data_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    
    target_needed = args.num_ollama - len(unused_records)
    if target_needed > 0 or args.force_generate:
        # We only need to generate the REMAINING examples
        producer = producer_class(target_needed, args.ollama_model, data_queue, stop_event)
        producer.start()
        console.print(f"[bold cyan][System] Producer active: Will distill {target_needed} new thoughts once JSONL is dry.[/bold cyan]")
    else:
        producer = None
        console.print(f"[bold green][System] Using {len(unused_records)} existing unused records. Producer idle.[/bold green]")
    
    console.print(f"[bold magenta]--- Starting Weight-Stealing Distillation ---[/bold magenta]")
    console.print(f"[dim]Targets: {args.num_ollama} from {args.ollama_model}[/dim]")
    
    nexus.set_mode("ACTIVE")
    i = 0
    try:
        while True:
            # Get next record (either from file or producer)
            if unused_records:
                record = unused_records.pop(0)
                source = "JSONL"
            else:
                try:
                    # Show queue status if we're waiting
                    q_size = data_queue.qsize()
                    if q_size == 0:
                        console.print("[dim]Waiting for Ollama to distill next thought...[/dim]")
                    
                    record = data_queue.get(timeout=10) # Wait up to 10s for teacher
                    if record is None: break # End of production
                    source = "LIVE"
                except queue.Empty:
                    if producer and not producer.is_alive(): break
                    if not producer: break # We ran out of local data and producer wasn't started
                    continue
                    
            prompt = record['prompt']
            response = record['response']
            
            # 1. VISUAL AUDIT (Denoising Snippet) - Optimized: Only every 10 steps
            if i % 10 == 0:
                console.print(f"\n[bold magenta]--- Step {i} [Source: {source} | Buffer: {data_queue.qsize()}] ---[/bold magenta]")
                prompt_ids = tokenizer.encode(prompt, pad=False)
                current_text = prompt_ids.unsqueeze(0).to(device)
                generated_ids = []
                
                with torch.no_grad():
                    for _ in range(15): # Shorter snippet
                        logits, _, _, _, _, _ = nexus.core(
                            text_input=current_text, diffusion_timesteps=1, # 1 step for speed
                            slow_state=nexus.ambient_field.slow_state,
                            mean_confidence=get_mean_confidence(nexus)
                        )
                        if logits is None: break
                        probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        if next_token.item() == tokenizer.eos_id: break
                        generated_ids.append(next_token.item())
                        current_text = torch.cat([current_text, next_token.unsqueeze(0)], dim=1)
                
                beast_snippet = tokenizer.decode(torch.tensor(generated_ids)).strip()
                console.print(f"\n[bold yellow]PROMPT:[/bold yellow] {prompt.replace('USER: ', '').replace('BEAST:', '').strip()[:100]}...")
                console.print(f"[bold green]OLLAMA:[/bold green] {response.replace('<|endoftext|>', '').strip()[:150]}...")
                console.print(f"[bold red]BEAST (Current Resonance):[/bold red] {beast_snippet}...")
            else:
                # Still show progress even without full audit
                if i % 1 == 0:
                    console.print(f"[dim]Step {i} | Source: {source} | Denoising...[/dim]", end="\r")
            
            # 2. PHYSICS-AWARE TRAINING STEP
            optimizer.zero_grad()
            
            full_text = prompt + response
            tokens = tokenizer.encode(full_text, pad=False).unsqueeze(0).to(device)
            
            with torch.no_grad():
                clean_latent = nexus.core.text_stack.integrate_input(tokens)
            
            # Diffusion training: sample random timestep, add noise, denoise
            # CE auxiliary loss keeps the lm_head projection sharp
            total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                nexus, clean_latent, max_T=12,
                ce_target=tokens, ce_weight=0.3
            )
            
            total_loss.backward()
            
            for p in nexus.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0)
            
            nexus.adjust_plasticity(optimizer)
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()
            
            # --- PHYSICS TICK ---
            nexus.core.broker.tick_decay()
            nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
            nexus.ambient_field.tick_consolidation()
            
            # 3. Dynamic Topology (Neurogenesis)
            structural_actions = nexus.neurogenesis.evaluate_topology()
            if structural_actions:
                nexus.neurogenesis.apply_structural_changes(structural_actions)
                nexus.core.shifter.refresh_registry()
            
            console.print(f"  [dim]Step {i} | MSE: {mse_val:.4f} | t={sampled_t} | noise={noise_scale:.3f}[/dim]")
            
            # 4. Periodic Saving (Air Optimizer)
            if i > 0 and i % 50 == 0:
                ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
                torch.save({'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, str(ckpt_path) + '.tmp')
                os.replace(str(ckpt_path) + '.tmp', str(ckpt_path))
                console.print(f"[dim green]Autosave: Checkpoint solidified at step {i}[/dim green]")
            
            i += 1
            if i >= args.num_ollama + 50: break

        # Loop finished naturally — save once
        console.print("[bold green]Training complete. Saving...[/bold green]")
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
        torch.save({'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, str(ckpt_path) + '.tmp')
        os.replace(str(ckpt_path) + '.tmp', str(ckpt_path))
        console.print("[green]Physical state calcified.[/green]")

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupt received. Saving current physics state...[/bold red]")
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
        torch.save({'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, str(ckpt_path) + '.tmp')
        os.replace(str(ckpt_path) + '.tmp', str(ckpt_path))
        console.print("[green]Physical state calcified. Safe to exit.[/green]")
    finally:
        stop_event.set()
        if producer is not None and producer.is_alive():
            producer.join(timeout=5)
    console.print(f"[bold green]Finished Automated Weight Distillation.[/bold green]")

def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"Initializing PCEI Instruction Tuner on {device}...")
    
    tokenizer = PCEITokenizer()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, args.layers_text, args.layers_img, args.layers_audio
    )
    console.print(f"Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    
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
    
    optimizer = torch.optim.AdamW(nexus.parameters(), lr=4e-4)
    
    # METABOLIC FLUSH: Increase plasticity for distillation
    if args.metabolic_flush:
        console.print("[bold red]Metabolic Flush: Decaying confidence to increase plasticity...[/bold red]")
        with torch.no_grad():
            for name, module in nexus.named_modules():
                if hasattr(module, 'slow_field_confidence'):
                    module.slow_field_confidence.data *= 0.5 # 50% decay to 'soften' the model
    if ckpt_path.exists():
        console.print(f"[green]Loading checkpoint from {ckpt_path.name}...[/green]")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except: pass
            
    # If --mode was not passed, show interactive menu
    if args.mode is None:
        args.mode = launch_prompt()

    if args.mode == "auto":
        sft_automated_mode(nexus, optimizer, tokenizer, args, device)
    elif args.mode == "basic":
        console.print("[bold green]--- Basic Conversational Mode ---[/bold green]")
        console.print("[dim]Strictly training on greetings, small talk, and human exchanges.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, producer_class=BasicOllamaProducer)
    elif args.mode == "chat":
        console.print("[bold magenta]--- Interactive RLHF Mode ---[/bold magenta]")
        pass

if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["auto", "basic", "chat"], default=None,
                        help="Training mode. If omitted, an interactive menu is shown.")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_micro.pt")
    parser.add_argument("--ollama-model", type=str, default="gpt-oss:120b-cloud")
    parser.add_argument("--num-ollama", type=int, default=100)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--metabolic-flush", action="store_true", help="Decay confidence to increase learning speed")
    parser.add_argument("--d-model-text", type=int, default=128)
    parser.add_argument("--d-model-img", type=int, default=128)
    parser.add_argument("--d-model-audio", type=int, default=64)
    parser.add_argument("--layers-text", type=int, default=2)
    parser.add_argument("--layers-img", type=int, default=1)
    parser.add_argument("--layers-audio", type=int, default=1)
    args = parser.parse_args()
    main(args)
