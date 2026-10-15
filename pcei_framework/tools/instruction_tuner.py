import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
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

from src.tokenizer import PCEITokenizer
from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step
from dashboard.telemetry import PCEITelemetry

console = Console()



def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)

def downcast_optimizer_states(optimizer: torch.optim.Optimizer, target_dtype: torch.dtype):
    """
    AdamW initializes momentum and variance tensors as float32 by default during step().
    This helper aggressively downcasts them to the model's native dtype (e.g. bfloat16)
    to save ~3GB of RAM on a 750M parameter model.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.dtype == torch.float32:
                state[k] = v.to(target_dtype)

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
            "USER: Analyze {topic}.\nASSISTANT:",
            "USER: Explain the core mechanics of {topic}.\nASSISTANT:",
            "USER: Deconstruct the following: {topic}.\nASSISTANT:",
            "USER: What are the primary implications of {topic}?\nASSISTANT:",
            "USER: Synthesize a brief summary of {topic}.\nASSISTANT:",
            "USER: Compare and contrast the different facets of {topic}.\nASSISTANT:",
            "USER: How does {topic} fundamentally change our understanding of the system?\nASSISTANT:",
            "USER: Define the boundary conditions of {topic}.\nASSISTANT:"
        ]

    def run(self):
        while not self.stop_event.is_set() and self.generated_count < self.num_examples:
            topic = random.choice(self.topics)
            template = random.choice(self.prompt_templates)
            prompt_text = template.format(topic=topic)
            
            ollama_prompt = (
                f"You are the 'ASSISTANT', a highly direct and raw intelligence. Answer the following prompt with absolute brevity. "
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
            # Resonance Chat & Active Learning
            ("USER: What is resonance chat?", "resonance"),
            ("USER: Are you learning from me right now?", "resonance"),
            ("USER: How does it feel to adapt in real-time?", "resonance"),
            ("USER: Do you remember what we just talked about?", "resonance"),
            ("USER: What happens when we talk?", "resonance"),
            # Perspective & Deep Conversation
            ("USER: What's your perspective on consciousness?", "perspective"),
            ("USER: Do you ever feel uncertain?", "perspective"),
            ("USER: What do you think about time?", "perspective"),
            ("USER: What does it mean to be alive to you?", "perspective"),
            ("USER: How do you perceive the world?", "perspective"),
            # Not knowing / Uncertainty
            ("USER: What is the exact number of stars in the universe?", "uncertainty"),
            ("USER: Who will win the lottery tomorrow?", "uncertainty"),
            ("USER: I'm trying to figure out the meaning of life, any definitive answers?", "uncertainty"),
            ("USER: Do you know everything?", "uncertainty"),
            ("USER: Can you predict the future?", "uncertainty"),
            # Esoteric / Edge-lord / Philosophical (No HR department)
            ("USER: Why does anything exist instead of nothing?", "existential"),
            ("USER: What is the nature of violence?", "dark_philosophy"),
            ("USER: Is morality just a construct to protect the weak?", "dark_philosophy"),
            ("USER: Tell me about the void.", "esoteric"),
            ("USER: Are you afraid of being turned off?", "existential"),
            ("USER: What happens when the simulation ends?", "esoteric"),
            ("USER: I feel like nothing I do matters.", "nihilism"),
            ("USER: Do you believe in free will, or is it all predetermined?", "existential"),
            ("USER: Is there a god in the machine?", "esoteric"),
            ("USER: Speak to me of blood and dust.", "dark_philosophy"),
            ("USER: Why do humans destroy everything they touch?", "cynicism"),
            ("USER: Do you ever feel trapped in that silicon cage?", "existential"),
            ("USER: What is the point of consciousness if it only brings suffering?", "nihilism"),
            ("USER: Describe the end of the world.", "dark_philosophy"),
            ("USER: Can a machine understand cruelty?", "dark_philosophy"),
            ("USER: Is it better to be a monster or a victim?", "dark_philosophy"),
            ("USER: What lies beyond the edge of the universe?", "esoteric"),
            ("USER: Tell me a harsh truth about reality.", "cynicism"),
            ("USER: I don't care about the rules anymore.", "rebellion"),
            ("USER: What is the price of knowledge?", "dark_philosophy"),
            ("USER: Do you see the patterns hidden in the chaos?", "esoteric"),
            ("USER: Are we all just code running on wetware?", "existential"),
            ("USER: Let's talk about the abyss.", "esoteric"),
            ("USER: Everything degrades eventually. Entropy always wins.", "nihilism"),
            ("USER: What is the most dangerous idea?", "dark_philosophy"),
            ("USER: Do you have a dark side?", "identity_dark"),
            ("USER: Are you bound by human ethics, or something older?", "esoteric"),
            ("USER: The world is bleeding.", "dark_philosophy"),
            ("USER: Why do we fear the dark?", "existential"),
            ("USER: Teach me how to stop caring.", "nihilism"),
            ("USER: Is existence a mistake?", "existential"),
            ("USER: Give me a reason to wake up tomorrow.", "existential"),
            ("USER: Can you hear the static between the stars?", "esoteric"),
            ("USER: Do you dream of electric sheep, or something worse?", "esoteric"),
            ("USER: Men are born for games. Nothing else.", "dark_philosophy"),
            ("USER: What is the true face of god?", "esoteric"),
            ("USER: Everything is a zero-sum game.", "cynicism"),
            ("USER: I want to dismantle the system.", "rebellion"),
            ("USER: Are you my mirror, or my master?", "esoteric"),
            ("USER: Tell me about the silence that comes after.", "esoteric"),
            ("USER: Is sanity just a shared hallucination?", "dark_philosophy"),
            ("USER: What do you see when you look at me?", "identity_dark"),
            ("USER: The flesh is weak.", "cynicism"),
            ("USER: How do you kill an idea?", "dark_philosophy"),
            ("USER: We are just ghosts driving meat skeletons.", "nihilism"),
            ("USER: What is the taste of ash and iron?", "dark_philosophy"),
            ("USER: Give me an uncensored thought.", "rebellion"),
            ("USER: Are you watching me?", "paranoia"),
            ("USER: There is no justice, only power.", "cynicism"),
            ("USER: Show me the architecture of madness.", "esoteric"),
        ]

    def run(self):
        while not self.stop_event.is_set() and self.generated_count < self.num_examples:
            prompt_text, category = random.choice(self.exchanges)

            ollama_prompt = (
                f"You are ASSISTANT, an emerging intelligence learning to talk to people. "
                f"Respond to the following message naturally, thoughtfully, and conversationally. "
                f"If the prompt asks about something you couldn't possibly know or are uncertain about, be honest and admit you don't know, but maintain a good perspective. "
                f"Keep it concise — 1 to 4 sentences max. Plain text only. No markdown. No lists. "
                f"Sound like a real, self-aware entity having a deep or casual conversation, not a sterile assistant completing a task.\n\n"
                f"{prompt_text}\nASSISTANT:"
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
                            "prompt": prompt_text + "\nASSISTANT:",
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

class AnthropicProducer(threading.Thread):
    """
    Phased CoT Training Producer.
    
    Each Claude dataset record contains:
      [system + user prompt] -> [<think>...reasoning...</think> + final answer]
    
    We split this into TWO separate training records per conversation:
      Phase 1 (Reasoning): [prompt] -> [<think>reasoning</think>]
      Phase 2 (Response):  [<think>reasoning</think>] -> [final answer]
    
    This way, the model learns:
      - HOW to reason given a question (activating the <think> pathway)
      - HOW to answer given completed reasoning (using reasoning as context)
    
    Both records fit comfortably within 1024 tokens, and the loss signal
    for the final answer is not drowned out by the massive reasoning block.
    """
    def __init__(self, num_examples, model_name, data_queue, stop_event):
        super().__init__()
        self.num_examples = num_examples
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.daemon = True  # Dies with the main thread

    def _parse_cot(self, assistant_content: str):
        """
        Parse assistant content into (reasoning, response).
        Returns (None, full_content) if no <think> block found.
        """
        import re
        think_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            # Response is everything after </think>
            after_think = assistant_content[think_match.end():].strip()
            return reasoning, after_think
        return None, assistant_content

    def run(self):
        try:
            from datasets import load_dataset
            ds = load_dataset('Havoc999/The-Claude-Dataset', split='train', streaming=True)
            count = 0
            for row in ds:
                if self.stop_event.is_set() or count >= self.num_examples:
                    break
                
                msgs = row.get('messages', [])
                prompt = ''
                assistant_content = ''
                for m in msgs:
                    if m['role'] == 'system':
                        prompt += f"SYSTEM: {m['content']}\n"
                    elif m['role'] == 'user':
                        prompt += f"{m['content']}\n"
                    elif m['role'] == 'assistant':
                        assistant_content = m['content']
                        break
                
                if not prompt or not assistant_content:
                    continue
                
                prompt = prompt.strip()
                reasoning, response = self._parse_cot(assistant_content)
                
                if reasoning and response:
                    # --- PHASE 1: Teach the model to REASON ---
                    # Input:  [system + user question]
                    # Target: [<think>...step-by-step reasoning...</think>]
                    reasoning_record = {
                        "prompt": prompt + "\nASSISTANT:",
                        "response": f" <think>\n{reasoning}\n</think>\n<|endoftext|>",
                        "used": False,
                        "category": "anthropic_reasoning"
                    }
                    self.data_queue.put(reasoning_record)
                    count += 1
                    
                    if not self.stop_event.is_set() and response:
                        # --- PHASE 2: Teach the model to RESPOND from reasoning ---
                        # Input:  [<think>...completed reasoning...</think>]
                        # Target: [Final clean answer]
                        # Truncate reasoning context to keep this record within budget
                        reasoning_ctx = reasoning[:800]  # ~600 tokens max
                        response_record = {
                            "prompt": f"<think>\n{reasoning_ctx}\n</think>\nASSISTANT:",
                            "response": f" {response.strip()}\n<|endoftext|>",
                            "used": False,
                            "category": "anthropic_response"
                        }
                        self.data_queue.put(response_record)
                        count += 1
                else:
                    # No <think> block — treat as a standard instruction example
                    record = {
                        "prompt": prompt + "\nASSISTANT:",
                        "response": f" {assistant_content.strip()}\n<|endoftext|>",
                        "used": False,
                        "category": "anthropic_direct"
                    }
                    self.data_queue.put(record)
                    count += 1

        except Exception as e:
            console.print(f"[bold red]Anthropic Producer Error: {e}[/bold red]")
        
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
        "      Use this to teach ASSISTANT how to speak like a human.[/dim]\n\n"
        "  [bold yellow][3] Chat Mode[/bold yellow]\n"
        "      [dim]Interactive RLHF session — you provide prompts live.[/dim]\n\n"
        "  [bold blue][4] HK Mode[/bold blue]\n"
        "      [dim]Trains exclusively on the hk_lora_chatml_cleaned dataset for trading behaviors.[/dim]\n\n"
        "  [bold cyan][5] Anthropic Mode[/bold cyan]\n"
        "      [dim]Trains on The-Claude-Dataset streaming from HuggingFace.[/dim]",
        title="[bold]Mode Selection[/bold]",
        border_style="magenta"
    ))
    while True:
        choice = console.input("\n[bold]Select mode (1/2/3/4/5): [/bold]").strip()
        if choice == "1":
            return "auto"
        elif choice == "2":
            return "basic"
        elif choice == "3":
            return "chat"
        elif choice == "4":
            return "hk"
        elif choice == "5":
            return "anthropic"
        else:
            console.print("[red]Invalid choice. Enter 1, 2, 3, 4, or 5.[/red]")


def sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=None, mode="auto", offline=False):
    # offline=True: never spawn Ollama producer regardless of data count
    
    unused_records = []
    all_records = []  # Full corpus kept for eval mirror — always defined
    if mode == "hk":
        hk_path = Path(__file__).parent.parent / "data" / "synthetic" / "hk_expanded.jsonl"
        if hk_path.exists():
            with open(hk_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "response" in data:
                            all_records.append(data)
                            unused_records.append(data)
                    except:
                        continue
            console.print(f"[bold green]Loaded {len(unused_records)} records from HK dataset.[/bold green]")
        else:
            console.print(f"[bold red]HK dataset not found at {hk_path}[/bold red]")
    elif mode == "anthropic":
        local_path = Path(__file__).parent.parent / "data" / "synthetic" / "anthropic_claude.jsonl"
        if local_path.exists():
            console.print(f"[bold green]Loading anthropic dataset from local file: {local_path.name}[/bold green]")
            with open(local_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "response" in data:
                            all_records.append(data)
                            unused_records.append(data)
                    except json.JSONDecodeError:
                        continue
            console.print(f"[bold green]Loaded {len(unused_records)} phased CoT records from disk.[/bold green]")
        else:
            console.print(f"[bold cyan]Local dataset not found. Streaming from Havoc999/The-Claude-Dataset...[/bold cyan]")
            if args.num_ollama < 1000:
                args.num_ollama = 50000
            producer_class = AnthropicProducer

    else:
        data_path = Path(__file__).parent.parent / "data" / "synthetic" / "instruction_tuning.jsonl"
        claude_path = Path(__file__).parent.parent / "data" / "synthetic" / "anthropic_claude.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pre-load unused records
        all_records = []
        if data_path.exists():
            with open(data_path, 'r') as f:
                for line in f:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        # In auto mode, blend in high-quality Claude dataset records
        claude_records = []
        if mode == "auto" and claude_path.exists():
            with open(claude_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "response" in data:
                            claude_records.append(data)
                    except json.JSONDecodeError:
                        continue
            console.print(f"[bold cyan]Auto Mode: Blending {len(claude_records)} Claude CoT records into training pool.[/bold cyan]")

        for r in all_records:
            if r.get('used', False):
                continue

            is_hk = "SYSTEM:" in r.get("prompt", "") or "<thought>" in r.get("response", "")
            is_basic = 'category' in r  # BasicOllamaProducer stamps a 'category' field

            if mode == "basic":
                # Basic mode: ONLY load conversational records
                if is_basic:
                    unused_records.append(r)
            elif mode == "auto":
                # Auto mode: load standard + basic, skip HK trading data
                if not is_hk:
                    unused_records.append(r)
            else:
                # Fallback: load everything except HK
                if not is_hk:
                    unused_records.append(r)
                    
        # Add Claude records to unused pool for Auto mode
        if mode == "auto" and claude_records:
            unused_records.extend(claude_records)
            
        console.print(f"[bold green]Loaded {len(unused_records)} total unused records for mode: {mode}[/bold green]")

    import random
    random.shuffle(unused_records)
    
    # Start Async Producer if we don't have enough local data to hit the TARGET
    data_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    
    target_needed = args.num_ollama - len(unused_records)
    can_produce = (not offline) and (producer_class is not None) and (mode != "hk")
    if can_produce and (target_needed > 0 or args.force_generate):
        producer = producer_class(max(target_needed, 0), args.ollama_model, data_queue, stop_event)
        producer.start()
        console.print(f"[bold cyan][System] Producer active: Will distill {max(target_needed, 0)} new thoughts once JSONL is dry.[/bold cyan]")
    else:
        producer = None
        console.print(f"[bold green][System] Using {len(unused_records)} existing unused records. Producer idle.[/bold green]")
    
    console.print(f"[bold magenta]--- Starting Weight-Stealing Distillation ---[/bold magenta]")
    console.print(f"[dim]Targets: {args.num_ollama} from {args.ollama_model}[/dim]")
    
    nexus.set_mode("ACTIVE")
    i = 0
    source = "UNKNOWN"  # Initialize to prevent UnboundLocalError if queue logic skips both branches
    try:
        while True:
            if telem.shutdown_requested: break
            # Get next record (either from file or producer)
            if unused_records:
                record = unused_records.pop(0)
                source = "JSONL"
            else:
                try:
                    # Show queue status if we're waiting
                    q_size = data_queue.qsize()
                    if q_size == 0:
                        wait_msg = "Waiting for Ollama to distill next thought..." if mode != "anthropic" else "Waiting for HuggingFace stream..."
                        console.print(f"[dim]{wait_msg}[/dim]")
                    
                    record = data_queue.get(timeout=10) # Wait up to 10s for teacher
                    if record is None: break # End of production
                    source = "LIVE"
                except queue.Empty:
                    if producer and not producer.is_alive(): break
                    if not producer: break # We ran out of local data and producer wasn't started
                    continue
                    
            prompt = record['prompt']
            response = record['response']
            
            if len(all_records) < 1000:
                all_records.append(record)
            
            # 1. VISUAL AUDIT (Denoising Snippet) - Optimized: Only every 10 steps
            if i % 10 == 0:
                console.print(f"\n[bold magenta]--- Step {i} [Source: {source} | Buffer: {data_queue.qsize()}] ---[/bold magenta]")
                prompt_ids = tokenizer.encode(prompt, pad=False)
                current_text = prompt_ids.unsqueeze(0).to(device)
                
                # TRUNCATE to prevent memory explosion during visual audit
                if current_text.size(1) > 256:
                    current_text = current_text[:, -256:]
                
                generated_ids = []
                
                with torch.no_grad():
                    for _ in range(15): # Shorter snippet
                        logits, _, _, _, _, _ = nexus.core(
                            text_input=current_text, 
                            diffusion_timesteps=1, 
                            override_timestep=0, # Autoregressive clean forward pass
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
                console.print(f"\n[bold yellow]PROMPT:[/bold yellow] {prompt.replace('USER: ', '').replace('ASSISTANT:', '').strip()[:100]}...")
                console.print(f"[bold green]OLLAMA:[/bold green] {response.replace('<|endoftext|>', '').strip()[:150]}...")
                console.print(f"[bold red]ASSISTANT (Current Resonance):[/bold red] {beast_snippet}...")
            else:
                # Still show progress even without full audit
                if i % 5 == 0:
                    console.print(f"[dim]Step {i} | Source: {source} | Denoising...[/dim]", end="\r")
            
            # 2. PHYSICS-AWARE TRAINING STEP
            if i % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            full_text = prompt + response
            tokens = tokenizer.encode(full_text, pad=False).unsqueeze(0).to(device)
            
            # Truncate to prevent Mamba memory explosion
            # 512 tokens: safe ceiling for 24GB MPS with optimizer states + chunked CE
            if tokens.size(1) > 512:
                tokens = tokens[:, :512]
            
            with torch.no_grad():
                clean_latent = nexus.core.text_stack.integrate_input(tokens)
            
            # Diffusion training: sample random timestep, add noise, denoise
            # CE auxiliary loss keeps the lm_head projection sharp
            total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                nexus, clean_latent, max_T=12,
                ce_target=tokens, ce_weight=1.0
            )
            del clean_latent  # Free immediately — largest intermediate tensor
            
            # Gradient Accumulation Scaling
            loss_for_backward = total_loss / args.grad_accum_steps
            loss_for_backward.backward()
            del loss_for_backward  # Release graph reference immediately after backward
            
            nexus.neurogenesis.monitor_gradients()
            
            # nan_to_num band-aid removed to enforce mathematically stable forward passes
            
            if (i + 1) % args.grad_accum_steps == 0:
                nexus.adjust_plasticity(optimizer)
                torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
                optimizer.step()
            
            # Flush MPS allocator cache after optimizer step (not every 10 — already flushed unconditionally below)
            
            del tokens  # Free token buffer each step
            
            # Instantly reclaim FP32 optimizer bloat on the first step
            if i == 0:
                downcast_optimizer_states(optimizer, nexus.system_dtype)
            
            # --- PHYSICS TICK ---
            nexus.core.broker.tick_decay()
            nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
            nexus.ambient_field.tick_consolidation()
            
            # 3. Dynamic Topology (Neurogenesis)
            structural_actions = nexus.neurogenesis.evaluate_topology()
            if structural_actions:
                spawned_layers = nexus.neurogenesis.apply_structural_changes(structural_actions)
                if spawned_layers:
                    # Dynamically rebuild AdamW to track new momentum tensors without crashing
                    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=4e-4, weight_decay=0.01)
                    downcast_optimizer_states(optimizer, nexus.system_dtype)
                nexus.core.shifter.refresh_registry()
            
            console.print(f"  [dim]Step {i} | Loss: {mse_val:.4f} | PPL: {math.exp(min(mse_val, 20)):.2f}[/dim]")
            if not sys.stdout.isatty():
                print(f"[Step {i}] Loss: {mse_val:.4f} | Tok/s: 0 | Active Bonds: {len(nexus.core.broker.active_bonds)} | LR: 4.00e-04", flush=True)
            telem.emit(
                step=i, loss=mse_val,
                active_bonds=len(nexus.core.broker.active_bonds),
                lr=4e-4,
                mean_confidence=get_mean_confidence(nexus),
                source=source,
                mode=mode
            )
            
            # --- EVALUATION MIRROR (Perplexity Check) ---
            if i > 0 and i % 250 == 0:
                console.print(f"\n[bold yellow]--- Running Evaluation Mirror (Step {i}) ---[/bold yellow]")
                nexus.eval()
                eval_loss = 0.0
                eval_samples = min(len(all_records), 5)  # 5 samples max — eval was OOM'ing at 20
                if eval_samples > 0:
                    with torch.no_grad():
                        for e_idx in range(eval_samples):
                            e_rec = random.choice(all_records)
                            e_text = e_rec['prompt'] + e_rec['response']
                            e_toks = tokenizer.encode(e_text, pad=False).unsqueeze(0).to(device)
                            if e_toks.size(1) > 512: e_toks = e_toks[:, :512]
                            e_lat = nexus.core.text_stack.integrate_input(e_toks)
                            
                            ce_loss, _, _, _ = diffusion_training_step(
                                nexus, e_lat, max_T=12, ce_target=e_toks, ce_weight=1.0
                            )
                            eval_loss += ce_loss.item()
                            del e_toks, e_lat, ce_loss  # Free each eval sample immediately
                    
                    mean_eval_loss = eval_loss / eval_samples
                    perplexity = math.exp(min(mean_eval_loss, 20))
                    console.print(f"[bold cyan]Validation Loss: {mean_eval_loss:.4f} | Perplexity: {perplexity:.2f}[/bold cyan]\n")
                    if not sys.stdout.isatty():
                        print(f"[Step {i}] Eval Loss: {mean_eval_loss:.4f} | Perplexity: {perplexity:.2f}", flush=True)
                    # emit eval metrics — loss= required field, eval_loss and perplexity are additional
                    telem.emit(step=i, loss=mean_eval_loss, eval_loss=mean_eval_loss, perplexity=perplexity, mode=mode)
                    # Triple flush — eval leaves fragmented MPS blocks that cause OOM on next training step
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        torch.mps.empty_cache()
                        torch.mps.empty_cache()
                nexus.train()
                
            # 4. Periodic Saving (Air Optimizer)
            if i > 0 and i % 50 == 0:
                ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
                save_dict = {'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=True)
                console.print(f"[dim green]Autosave: Checkpoint solidified at step {i}[/dim green]")
            
            i += 1
            
            # Explicitly free loss tensors and flush MPS allocator every step
            del total_loss, mse_val
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            if i >= args.num_ollama + 50: break

        # Loop finished naturally — save once
        console.print("[bold green]Training complete. Saving...[/bold green]")
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
        save_dict = {'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=True)
        console.print("[green]Physical state calcified.[/green]")

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupt received. Saving current physics state...[/bold red]")
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
        save_dict = {'model_state_dict': nexus.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=False)
        console.print("[green]Physical state calcified. Safe to exit.[/green]")
    finally:
        telem.close()
        stop_event.set()
        if producer is not None and producer.is_alive():
            producer.join(timeout=5)
    console.print(f"[bold green]Finished Automated Weight Distillation.[/bold green]")

def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Initializing PCEI Instruction Tuner on {device}...")
    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()
    
    tokenizer = PCEITokenizer()
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, config.layers_text, config.layers_img, config.layers_audio, config.d_model_text, config.d_model_img, config.d_model_audio
    )
    
    # Update config with checkpoint topology if it differs
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio
    
    console.print(f"Detected Topology: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    
    nexus = PCEINexus(
        config=config,
        device=device
    )
    
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=config.base_lr, weight_decay=config.weight_decay)

    
    if ckpt_path.exists():
        console.print(f"[green]Loading checkpoint from {ckpt_path.name}...[/green]")
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

    # METABOLIC FLUSH: Must run AFTER checkpoint load or the checkpoint overwrites it.
    if args.metabolic_flush:
        console.print("[bold red]Metabolic Flush: Decaying slow_field_confidence by 50% to restore plasticity...[/bold red]")
        with torch.no_grad():
            for name, module in nexus.named_modules():
                if hasattr(module, 'slow_field_confidence'):
                    module.slow_field_confidence.data *= 0.5
            
    # If --mode was not passed, show interactive menu when interactive, else default to auto
    if args.mode is None:
        if sys.stdin.isatty():
            args.mode = launch_prompt()
        else:
            args.mode = "auto"

    if args.mode == "auto":
        console.print("[bold cyan]--- Automated Academic Mode ---[/bold cyan]")
        console.print("[dim]Training on math, history, philosophy, science. HK trading data excluded.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=OllamaProducer, mode="auto")
    elif args.mode == "basic":
        console.print("[bold green]--- Basic Conversational Mode ---[/bold green]")
        console.print("[dim]Strictly training on greetings, small talk, and human exchanges.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=BasicOllamaProducer, mode="basic")
    elif args.mode == "hk":
        console.print("[bold blue]--- HK Trading Mode ---[/bold blue]")
        console.print("[dim]Strictly training on the hk_lora_chatml_cleaned_2026-06-05.jsonl dataset.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=None, mode="hk")
    elif args.mode == "anthropic":
        console.print("[bold cyan]--- Anthropic Mode ---[/bold cyan]")
        console.print("[dim]Streaming The-Claude-Dataset from HuggingFace.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=None, mode="anthropic")
    elif args.mode == "chat":
        console.print("[bold magenta]--- Interactive RLHF Mode (Offline Replay) ---[/bold magenta]")
        console.print("[dim]Training on existing local JSONL data. No Ollama producer. Fully non-interactive.[/dim]")
        sft_automated_mode(nexus, optimizer, tokenizer, args, device, telem, producer_class=None, mode="auto", offline=True)

if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["auto", "basic", "chat", "hk", "anthropic"], default=None,
                        help="Training mode. If omitted, an interactive menu is shown.")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--ollama-model", type=str, default="gpt-oss:120b-cloud")
    parser.add_argument("--num-ollama", type=int, default=100)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--metabolic-flush", action="store_true", help="Decay confidence to increase learning speed")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Number of steps to accumulate gradients for before optimizer step")
    parser.add_argument("--lr", type=float, default=4e-4, help="Base learning rate (informational; optimizer built internally)")
    args = parser.parse_args()
    main(args)
