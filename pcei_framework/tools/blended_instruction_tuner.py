#!/usr/bin/env python3
"""
tools/blended_instruction_tuner.py

Unified, Blended Instruction & Deep Reasoning Tuner for PCEI.
TOKEN-MAXX EDITION: Continuous Sequence Packing + Multi-Stream Ingestion + TF32 Acceleration.

Curriculum (7.2 GB Available):
  - 35% OpenHermes 2.5 (Deductive logic, multi-step math, Python coding, CoT)
  - 20% UltraChat 200k (Multi-paragraph natural dialogue & deep explanations)
  - 15% Anthropic Claude (Chain-of-thought analysis & structured reasoning)
  -  5% Anthropic HH-RLHF (Helpfulness & nuanced conversation)
  - 10% FineWeb-Edu (Educational STEM rehearsal anchor)
  -  8% Cosmopedia (High-density synthetic textbooks)
  -  4% TinyStories (Commonsense world reasoning)
  -  3% Classical Philosophy & Modern Literature (Plato, Nietzsche, Infinite Jest)

Features:
  - Continuous sequence packing: Zero padding waste; packs multiple passages per 512-token chunk.
  - Strict prompt-masking: Cross-entropy is evaluated strictly on assistant target tokens.
  - Configurable micro-batch size (1 for local Mac MPS/CPU, 2 for 24GB CUDA).
  - TF32 matrix core acceleration on Ampere/Ada Lovelace GPUs.
  - Cosine learning rate schedule with warmup (1.0e-4 -> 1.0e-5).
  - Dynamic physics updates (PlasticLayer confidence, active bonds decay, ambient consolidation).
  - Live generation probes every N steps to monitor text reasoning evolution.
  - Credit-aware safe checkpoint saving every 10k steps and at $2.00 credit floor.
"""

import os
import sys
import math
import time
import json
import random
import re
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step


PROBE_PROMPTS = [
    {
        "domain": "Identity & Persona",
        "prompt": "USER: Hello! What is your name and what model architecture are you?\nASSISTANT:",
        "max_new": 90,
    },
    {
        "domain": "Elementary Arithmetic",
        "prompt": "USER: What is 2 + 2, and what is 15 - 8?\nASSISTANT:",
        "max_new": 60,
    },
    {
        "domain": "Deductive Logic & Reasoning",
        "prompt": "USER: Read these premises:\n1. All mammals are warm-blooded.\n2. All dolphins are mammals.\n3. Flipper is a dolphin.\nWhat can we conclude about Flipper? Explain your reasoning step by step.\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "Scientific Explanation",
        "prompt": "USER: Why does ice float on liquid water, and why is this property essential for aquatic life in winter?\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "Python Programming",
        "prompt": "USER: Write a clean Python function to find the maximum value in a list of numbers without using the built-in max() function.\nASSISTANT:",
        "max_new": 140,
    },
]


def parse_conversation_turns(text: str) -> List[Tuple[str, str]]:
    """Extracts (role, content) turns from text formatted with USER:/ASSISTANT: or Human:/Assistant: or Prompt:/Response:"""
    normalized = text.replace("\n\nHuman:", "\nUSER:").replace("Human:", "USER:")
    normalized = normalized.replace("\n\nAssistant:", "\nASSISTANT:").replace("Assistant:", "ASSISTANT:")
    normalized = normalized.replace("**Prompt:**", "USER:").replace("\n\nPrompt:", "\nUSER:").replace("Prompt:", "USER:")
    normalized = normalized.replace("**Response:**", "ASSISTANT:").replace("\n\nResponse:", "\nASSISTANT:").replace("Response:", "ASSISTANT:")
    parts = re.split(r"(USER:|ASSISTANT:)", normalized)
    turns = []
    curr_role = None
    for p in parts:
        p_strip = p.strip()
        if p_strip == "USER:":
            curr_role = "user"
        elif p_strip == "ASSISTANT:":
            curr_role = "assistant"
        elif curr_role and p.strip():
            turns.append((curr_role, p.strip()))
    return turns


class BlendedDatasetStreamer:
    """
    Infinite streaming iterator that interleaves multiple datasets
    (JSONL, ShareGPT JSON, and raw text files) with multi-turn alternating
    loss masking and continuous sequence packing up to max_seq_len.
    """
    def __init__(self, dataset_specs: List[Dict], tokenizer: PCEITokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset_specs = dataset_specs
        self.active_sources = []
        self.weights = []

        self.token_buffer: List[int] = []
        self.label_buffer: List[int] = []
        self.buffer_source: Optional[str] = None

        for spec in dataset_specs:
            p = Path(spec["path"])
            if p.exists() and p.stat().st_size > 0:
                source_entry = {
                    "name": spec["name"],
                    "path": p,
                    "type": spec["type"],
                    "weight": spec["weight"],
                }
                if spec["type"] == "sharegpt_json":
                    try:
                        print(f"[DataStreamer] Loading ShareGPT JSON into memory from {p.name}...")
                        t0 = time.time()
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            source_entry["data"] = json.load(f)
                        random.shuffle(source_entry["data"])
                        source_entry["idx"] = 0
                        print(f"[DataStreamer] Loaded {len(source_entry['data'])} ShareGPT conversations in {time.time()-t0:.2f}s.")
                    except Exception as e:
                        print(f"[DataStreamer] Error loading ShareGPT JSON ({e}), skipping.")
                        continue
                else:
                    source_entry["file_obj"] = open(p, "r", encoding="utf-8", errors="replace")

                self.active_sources.append(source_entry)
                self.weights.append(spec["weight"])
                print(f"[DataStreamer] Active source: {spec['name']} ({p.name}) | Weight: {spec['weight']}")
            else:
                print(f"[DataStreamer] Notice: Optional dataset not found, skipping: {p.name}")

        if not self.active_sources:
            raise RuntimeError("No valid dataset sources found! Check dataset paths.")

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

    def _read_next_line(self, src: Dict) -> str:
        f = src["file_obj"]
        line = f.readline()
        if not line:
            try:
                f.seek(0)
                line = f.readline()
            except Exception:
                pass
            if not line:
                try:
                    f.close()
                    src["file_obj"] = open(src["path"], "r", encoding="utf-8", errors="replace")
                    line = src["file_obj"].readline()
                except Exception:
                    pass
        return line

    def _get_single_passage(self) -> Tuple[List[Tuple[str, str]], str]:
        """Returns (turns, source_name) where turns is a list of (role, text) tuples."""
        for _ in range(30):
            src_idx = random.choices(range(len(self.active_sources)), weights=self.weights, k=1)[0]
            src = self.active_sources[src_idx]
            src_type = src["type"]

            if src_type == "sharegpt_json":
                data_list = src["data"]
                item = data_list[src["idx"]]
                src["idx"] = (src["idx"] + 1) % len(data_list)
                turns = []
                for conv in item.get("conversations", []):
                    role = conv.get("from", "").lower().strip()
                    val = conv.get("value", "").strip()
                    if not val:
                        continue
                    if role in ("human", "user"):
                        turns.append(("user", val))
                    elif role in ("gpt", "chatgpt", "bing", "assistant"):
                        turns.append(("assistant", val))
                    elif role == "system":
                        turns.append(("system", val))
                if turns:
                    return turns, src["name"]
                continue

            elif src_type == "text_file":
                line = self._read_next_line(src).strip()
                if len(line) < 30:
                    continue
                return [("raw", line)], src["name"]

            line = self._read_next_line(src).strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            if src_type == "conversation_text":
                if "conversations" in data:
                    turns = []
                    for conv in data["conversations"]:
                        role = conv.get("from", "").lower().strip()
                        val = conv.get("value", "").strip()
                        if not val:
                            continue
                        if role in ("human", "user"):
                            turns.append(("user", val))
                        elif role in ("gpt", "chatgpt", "bing", "assistant"):
                            turns.append(("assistant", val))
                        elif role == "system":
                            turns.append(("system", val))
                    if turns:
                        return turns, src["name"]

                raw_text = data.get("text", "")
                parsed = parse_conversation_turns(raw_text)
                if parsed:
                    return parsed, src["name"]
                elif len(raw_text) > 30:
                    return [("raw", raw_text)], src["name"]

            elif src_type == "prompt_response":
                p = data.get("prompt", "").strip()
                r = data.get("response", "").strip()
                if p and r:
                    if p.endswith("ASSISTANT:"):
                        p = p[:-len("ASSISTANT:")].strip()
                    if p.startswith("USER:"):
                        p = p[len("USER:"):].strip()
                    return [("user", p), ("assistant", r)], src["name"]

            elif src_type == "raw_text":
                raw = data.get("text", "").strip()
                if len(raw) >= 20:
                    return [("raw", raw)], src["name"]

        # Fallback
        return [("user", "What architecture is PCEI?"),
                ("assistant", "PCEI is a physical continuum emergent intelligence with ternary BitNet weights and Mamba state-space blocks.")], "Fallback"

    def _turns_to_tokens_and_labels(self, turns: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
        tokens = []
        labels = []
        for role, content in turns:
            if not content:
                continue
            if role == "raw":
                t_ids = self.tokenizer.encode(content + "<|endoftext|>\n", pad=False).tolist()
                tokens.extend(t_ids)
                labels.extend(t_ids)
            elif role in ("user", "system"):
                prefix = "USER: " if role == "user" else "SYSTEM: "
                text = f"{prefix}{content.strip()}\nASSISTANT:"
                t_ids = self.tokenizer.encode(text, pad=False).tolist()
                tokens.extend(t_ids)
                labels.extend([0] * len(t_ids))  # Mask out prompt tokens
            elif role == "assistant":
                text = f" {content.strip()}<|endoftext|>\n"
                t_ids = self.tokenizer.encode(text, pad=False).tolist()
                tokens.extend(t_ids)
                labels.extend(t_ids)  # Active target loss
        return tokens, labels

    def get_packed_sample(self, pack: bool = True) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Returns (tokens, labels, source_name) with continuous sequence packing up to max_seq_len."""
        primary_source = self.buffer_source or "Blended"

        while len(self.token_buffer) < self.max_seq_len:
            turns, src_name = self._get_single_passage()
            if not self.buffer_source:
                self.buffer_source = src_name
            toks, labs = self._turns_to_tokens_and_labels(turns)
            self.token_buffer.extend(toks)
            self.label_buffer.extend(labs)
            if not pack:
                break

        if len(self.token_buffer) >= self.max_seq_len:
            sample_tokens = self.token_buffer[:self.max_seq_len]
            sample_labels = self.label_buffer[:self.max_seq_len]
            self.token_buffer = self.token_buffer[self.max_seq_len:]
            self.label_buffer = self.label_buffer[self.max_seq_len:]
            primary_source = self.buffer_source or "Blended"
            if len(self.token_buffer) == 0:
                self.buffer_source = None
        else:
            pad_len = self.max_seq_len - len(self.token_buffer)
            sample_tokens = self.token_buffer + [0] * pad_len
            sample_labels = self.label_buffer + [0] * pad_len
            self.token_buffer.clear()
            self.label_buffer.clear()
            primary_source = self.buffer_source or "Blended"
            self.buffer_source = None

        tokens_tensor = torch.tensor(sample_tokens, dtype=torch.long)
        labels_tensor = torch.tensor(sample_labels, dtype=torch.long)

        # Safety guard: ensure at least some unmasked labels exist
        if (labels_tensor != 0).sum().item() == 0:
            half = len(labels_tensor) // 2
            labels_tensor[half:] = tokens_tensor[half:]

        return tokens_tensor, labels_tensor, primary_source

    def get_batch(self, batch_size: int = 1, pack: bool = True) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Returns a batched tensor pair (tokens, labels) of shape (batch_size, max_seq_len)."""
        batch_tokens = []
        batch_labels = []
        sources = []
        for _ in range(batch_size):
            toks, labs, src = self.get_packed_sample(pack=pack)
            batch_tokens.append(toks)
            batch_labels.append(labs)
            sources.append(src)
        return torch.stack(batch_tokens), torch.stack(batch_labels), sources[0]

    def close(self):
        for src in self.active_sources:
            if "file_obj" in src:
                try:
                    src["file_obj"].close()
                except Exception:
                    pass


def run_probe_generation(nexus: PCEINexus, tokenizer: PCEITokenizer, device: str) -> None:
    """Runs a live generation test on fixed benchmark prompts to evaluate reasoning progress."""
    nexus.eval()
    print("\n" + "=" * 70)
    print("  [LIVE GENERATION PROBE - REASONING & CONVERSATION CHECK]")
    print("=" * 70)

    # Parasitic junk tokens ban
    bad_ids = set(range(94, 256))
    junk_patterns = [
        'reportprint', 'rawdownload', 'RandomRedditor', 'TheNitrome',
        'externalTo', 'oreAndOnline', 'quickShip', 'StreamerBot',
        'ActionCode', 'RecipeTag', 'SubmitButton', 'InstoreAndOnline',
        'DiaryInfo', 'ÃÂ', 'DeliveryDate', 'Nitrome', 'ThumbnailImage',
        'oreAnd', 'Diary', 'Recipe', 'Recipes', '="', '=\"', '</', '/>', '><'
    ]
    for i in range(tokenizer.vocab_size):
        try:
            s = tokenizer.enc.decode([i])
            if any(p in s for p in junk_patterns) or any(ord(c) > 1000 for c in s) or (len(s) == 1 and ord(s) < 32 and s not in '\n\t '):
                bad_ids.add(i)
        except Exception:
            bad_ids.add(i)
    bad_tensor = torch.tensor(list(bad_ids), device=device)

    syntax_words = [
        ' the', ' of', ' and', ' to', ' in', ' a', ' is', ' that', ' for', ' on',
        ' with', ' as', ' by', ' at', ' from', ' it', ' an', ' be', ' was', ' are',
        '.', ',', ' ', '\n', '-', ':', '(', ')', '"', "'", '!', '?', ';'
    ]
    syntax_ids = set()
    for w in syntax_words:
        for t in tokenizer.encode(w, pad=False).tolist():
            syntax_ids.add(t)

    def get_banned_3g(gen_ids):
        if len(gen_ids) < 2:
            return []
        prefix = tuple(gen_ids[-2:])
        banned = []
        for i in range(len(gen_ids) - 2):
            if tuple(gen_ids[i:i + 2]) == prefix:
                banned.append(gen_ids[i + 2])
        return banned

    for item in PROBE_PROMPTS:
        domain = item["domain"]
        prompt = item["prompt"]
        max_new = item["max_new"]

        print(f"\n--- {domain} ---")
        print(f"PROMPT: {prompt.strip()}")
        print("ASSISTANT: ", end="", flush=True)

        ids = tokenizer.encode(prompt, pad=False).unsqueeze(0).to(device)
        curr = ids.clone()
        generated_tokens = []
        stop_phrases = ["\nUSER:", "\nHuman:", "<|endoftext|>", "USER:", "\nUSER"]
        accumulated_text = ""

        with torch.no_grad():
            for _ in range(max_new):
                logits, *_ = nexus.core(
                    text_input=curr,
                    diffusion_timesteps=1,
                    override_timestep=0,
                    slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
                )
                logits_last = logits[:, -1, :].squeeze(0).clone().float()

                # 1. Mask junk tokens
                logits_last[bad_tensor] = -1e4

                # 2. Ban 3-gram repeats
                banned_3g = get_banned_3g(generated_tokens)
                if banned_3g:
                    logits_last[torch.tensor(banned_3g, device=device)] = -1e4

                # 3. Non-syntax consecutive repeat ban
                if len(generated_tokens) > 0:
                    last_tok = generated_tokens[-1]
                    if last_tok not in syntax_ids:
                        logits_last[last_tok] = -1e4

                # 4. Additive frequency penalty
                if len(generated_tokens) > 0:
                    recent = generated_tokens[-40:]
                    counts = torch.bincount(torch.tensor(recent, device=device), minlength=tokenizer.vocab_size)
                    repeat_mask = (counts >= 2)
                    for s_id in syntax_ids:
                        if s_id < len(repeat_mask):
                            repeat_mask[s_id] = False
                    logits_last[repeat_mask] -= 4.0

                # 5. Multiplicative repetition penalty
                if len(generated_tokens) > 0:
                    recent_unique = set(generated_tokens[-40:])
                    for t in recent_unique:
                        if t not in syntax_ids:
                            if logits_last[t] > 0:
                                logits_last[t] /= 1.15
                            else:
                                logits_last[t] *= 1.15

                scaled = logits_last / 0.4
                probs = F.softmax(scaled, dim=-1)
                sprobs, sidx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sprobs, dim=-1)
                remove_mask = cum - sprobs > 0.85
                probs[sidx[remove_mask]] = 0.0
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                    next_tok = torch.multinomial(probs, 1).item()
                else:
                    next_tok = sidx[0].item()

                if next_tok in (0, 50256):
                    break

                generated_tokens.append(next_tok)
                piece = tokenizer.decode([next_tok])
                accumulated_text += piece
                sys.stdout.write(piece)
                sys.stdout.flush()

                if any(sp in accumulated_text for sp in stop_phrases):
                    break

                curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)

        print("\n" + "-" * 40)

    print("=" * 70 + "\n")
    nexus.train()


def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)


def downcast_optimizer_states(optimizer: torch.optim.Optimizer, target_dtype: torch.dtype):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.dtype == torch.float32:
                state[k] = v.to(target_dtype)


def check_credit_guard(log_path: Path = Path("/workspace/credit_guard.log"), min_credit: float = 2.0) -> bool:
    """Returns True if training should halt to preserve credit balance."""
    if not log_path.exists():
        return False
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines[-15:]):
            if "Current balance: $" in line:
                balance_str = line.split("Current balance: $")[-1].strip()
                balance = float(balance_str)
                if balance <= min_credit:
                    print(f"\n[CreditGuard Alert] Balance is ${balance:.2f} <= ${min_credit:.2f}. Triggering safety stop!")
                    return True
                break
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Blended Instruction & Reasoning Tuner for PCEI")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_750m_reasoning_assistant_latest.pt",
                        help="Base checkpoint to train from (e.g. pcei_750m_reasoning_assistant_latest.pt)")
    parser.add_argument("--output-name", type=str, default="pcei_750m_reasoning_assistant.pt",
                        help="Name of final aligned checkpoint")
    parser.add_argument("--steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size (1 for 2048 ctx)")
    parser.add_argument("--pack-sequences", action="store_true", default=True, help="Enable continuous sequence packing")
    parser.add_argument("--base-lr", type=float, default=5.0e-5, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1.0e-5, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (2048 for multi-turn)")
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between telemetry log")
    parser.add_argument("--probe-every", type=int, default=2500, help="Steps between live generation probes")
    parser.add_argument("--save-every", type=int, default=10000, help="Steps between intermediate checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing dataset jsonl files")
    parser.add_argument("--curriculum-mode", type=str, default="blended", choices=["blended", "precise_knowledge", "conversational_reasoning", "targeted_annealing", "kindergarten", "pcei_grounding"],
                        help="Curriculum mode: 'blended', 'precise_knowledge', 'conversational_reasoning', 'targeted_annealing', 'kindergarten', or 'pcei_grounding'")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda, mps, cpu)")
    args = parser.parse_args()

    # Determine Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Enable TF32 matrix core acceleration on Ampere/Ada Lovelace
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    effective_tokens_per_step = args.batch_size * args.max_seq_len * args.grad_accum_steps
    print(f"================================================================================")
    print(f"  PCEI TOKEN-MAXX INSTRUCTION & REASONING TUNER")
    print(f"  Target Device: {device.upper()} | Base Checkpoint: {args.checkpoint_name}")
    print(f"  Micro-Batch: {args.batch_size} | Accum: {args.grad_accum_steps} | Eff Batch: {effective_tokens_per_step} tok/opt_step")
    print(f"  Continuous Packing: {'ENABLED' if args.pack_sequences else 'DISABLED'} | TF32: {'ENABLED' if device == 'cuda' else 'DISABLED'}")
    print(f"  Target Steps: {args.steps} | LR: {args.base_lr:.2e} -> {args.min_lr:.2e}")
    print(f"================================================================================")

    # Initialize Tokenizer and Config
    tokenizer = PCEITokenizer()
    config = PCEIConfig.load()
    ckpt_path = PROJECT_ROOT / "checkpoints" / args.checkpoint_name

    # Fallback to base pretrain if latest doesn't exist
    if not ckpt_path.exists():
        fallback_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_dreamed_base.pt"
        if fallback_path.exists():
            print(f"[Checkpoint] {ckpt_path.name} not found, falling back to {fallback_path.name}...")
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load Topology from Checkpoint
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio

    print(f"[Topology] {layers_text} Text Layers (D={d_text}) | {layers_img} Img | {layers_audio} Audio")

    # Instantiate Model
    nexus = PCEINexus(config=config, device=device)

    # Load State Dict
    print(f"[Checkpoint] Loading weights from {ckpt_path.name}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
    nexus.load_ambient_state(checkpoint)
    nexus.to(nexus.system_dtype)

    # Build optimizer AFTER model parameters are fully loaded and cast to system_dtype
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.base_lr, weight_decay=config.weight_decay)

    # Re-restore broker active bonds if saved
    if 'active_bonds' in checkpoint and hasattr(nexus.core, 'broker'):
        nexus.core.broker.active_bonds = checkpoint['active_bonds']
        print(f"[Broker] Restored {len(nexus.core.broker.active_bonds)} active bonds.")

    nexus.set_mode("ACTIVE")
    nexus.train()

    # Enable gradient checkpointing on TextStack to keep activation memory under 1 GB at 2048 seq len
    if hasattr(nexus.core.text_stack, 'enable_gradient_checkpointing'):
        nexus.core.text_stack.enable_gradient_checkpointing()
        print("[Memory] Gradient checkpointing ENABLED on TextStack.")

    # Graceful shutdown handler
    shutdown_requested = False
    def sig_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n[Signal] Termination requested. Finishing current step and saving checkpoint...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Setup Dataset Streamer with Expanded Curriculum
    data_dir = PROJECT_ROOT / args.data_dir
    text_dir = PROJECT_ROOT / "data" / "raw" / "text"
    philosophy_dir = text_dir / "philosophy"
    modern_dir = text_dir / "modern_books"

    teacher_curriculum = data_dir / "pcei_foundation_teacher.jsonl"
    if not teacher_curriculum.exists():
        teacher_curriculum = PROJECT_ROOT / "data" / "curriculum" / "pcei_foundation_teacher.jsonl"

    synthetic_dir = data_dir if (data_dir / "beast_reasoning.jsonl").exists() else (data_dir / "synthetic")
    if not (synthetic_dir / "beast_reasoning.jsonl").exists():
        synthetic_dir = PROJECT_ROOT / "data" / "synthetic"

    if args.curriculum_mode == "pcei_grounding":
        print("[Curriculum] PCEI GROUNDING MODE: 50% PCEI Teacher (identity, physics, architecture) + 30% Procedural Arithmetic + 20% TinyStories (narrative rehearsal). Zero fluff.")
        arithmetic_path = PROJECT_ROOT / "data" / "curriculum" / "kindergarten_arithmetic.jsonl"
        tinystories_path = PROJECT_ROOT / "data" / "raw" / "text" / "mass_reasoning" / "tinystories_bulk.txt"
        dataset_specs = [
            {"name": "PCEI-Teacher",      "path": teacher_curriculum,"type": "prompt_response",  "weight": 0.50},
            {"name": "Kindergarten-Math", "path": arithmetic_path,   "type": "prompt_response",  "weight": 0.30},
            {"name": "TinyStories",       "path": tinystories_path,  "type": "text_file",       "weight": 0.20},
        ]
    elif args.curriculum_mode == "kindergarten":
        print("[Curriculum] KINDERGARTEN MODE: 60% TinyStories + 30% Procedural Arithmetic + 10% PCEI Teacher. Zero chat fluff. Zero complex STEM.")
        arithmetic_path = PROJECT_ROOT / "data" / "curriculum" / "kindergarten_arithmetic.jsonl"
        tinystories_path = PROJECT_ROOT / "data" / "raw" / "text" / "mass_reasoning" / "tinystories_bulk.txt"
        dataset_specs = [
            {"name": "TinyStories",       "path": tinystories_path,  "type": "text_file",       "weight": 0.60},
            {"name": "Kindergarten-Math", "path": arithmetic_path,   "type": "prompt_response",  "weight": 0.30},
            {"name": "PCEI-Teacher",      "path": teacher_curriculum,"type": "prompt_response",  "weight": 0.10},
        ]
    elif args.curriculum_mode == "precise_knowledge":
        print("[Curriculum] PRECISE KNOWLEDGE MODE: Purging conversational fluff, anchoring 100% on PCEI architecture, logic, concepts, and STEM facts.")
        dataset_specs = [
            # Foundational Teacher Anchor (35%) — PCEI Identity, Continuum, BitNet, Mamba Ground Truth
            {"name": "PCEI-Teacher",               "path": teacher_curriculum,                                              "type": "prompt_response",   "weight": 0.35},
            # Deductive Relational Syllogisms & Logic Anchor (25%)
            {"name": "Beast-Deductive-Reasoning",  "path": synthetic_dir / "beast_reasoning.jsonl",                          "type": "conversation_text", "weight": 0.25},
            # Concept-Attribute Relational Anchor (15%)
            {"name": "Congruent-Concepts",         "path": synthetic_dir / "congruent_concepts.jsonl",                      "type": "raw_text",          "weight": 0.15},
            # Knowledge Rehearsal STEM Anchor (25%) — Deep academic / science facts from FineWeb
            {"name": "FineWeb-Edu",                "path": data_dir / "fineweb_edu.jsonl",                                   "type": "raw_text",          "weight": 0.25},
        ]
    elif args.curriculum_mode == "conversational_reasoning":
        print("[Curriculum] CONVERSATIONAL REASONING MODE: 100% structured dialogue (PCEI-Teacher + Claude CoT + Beast Logic + Congruent Concepts), zero web junk.")
        dataset_specs = [
            # Foundational Teacher Anchor (35%) — PCEI Identity, Continuum, BitNet, Mamba Ground Truth
            {"name": "PCEI-Teacher",               "path": teacher_curriculum,                                              "type": "prompt_response",   "weight": 0.35},
            # Anthropic Claude Chain-of-Thought & Reasoning (30%)
            {"name": "Claude-CoT",                 "path": data_dir / "anthropic_claude.jsonl",                              "type": "prompt_response",   "weight": 0.30},
            # Deductive Relational Syllogisms & Logic Anchor (20%)
            {"name": "Beast-Deductive-Reasoning",  "path": synthetic_dir / "beast_reasoning.jsonl",                          "type": "conversation_text", "weight": 0.20},
            # Concept-Attribute Relational Anchor (15%)
            {"name": "Congruent-Concepts",         "path": synthetic_dir / "congruent_concepts.jsonl",                      "type": "raw_text",          "weight": 0.15},
        ]
    elif args.curriculum_mode == "targeted_annealing":
        print("[Curriculum] TARGETED ANNEALING MODE: 50% High-Density Exemplars (Arithmetic CoT, Syllogisms, Density Physics, Python, Persona) + 50% Claude-CoT general anchor.")
        exemplars_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_reasoning_exemplars.jsonl"
        if not exemplars_path.exists():
            exemplars_path = data_dir / "pcei_reasoning_exemplars.jsonl"
        dataset_specs = [
            # High-Density Targeted Reasoning Exemplars (50%)
            {"name": "Reasoning-Exemplars",        "path": exemplars_path,                                                  "type": "prompt_response",   "weight": 0.50},
            # Anthropic Claude Chain-of-Thought & General Reasoning (50%)
            {"name": "Claude-CoT",                 "path": data_dir / "anthropic_claude.jsonl",                              "type": "prompt_response",   "weight": 0.50},
        ]
    else:
        dataset_specs = [
            # Anthropic Claude Chain-of-Thought & Deep Analysis (25%)
            {"name": "Claude-CoT",                 "path": data_dir / "anthropic_claude.jsonl",                              "type": "prompt_response",   "weight": 0.25},
            # Deductive Relational Syllogisms & Logic Anchor (12%)
            {"name": "Beast-Deductive-Reasoning",  "path": synthetic_dir / "beast_reasoning.jsonl",                          "type": "conversation_text", "weight": 0.12},
            # Concept-Attribute Relational Anchor (10%)
            {"name": "Congruent-Concepts",         "path": synthetic_dir / "congruent_concepts.jsonl",                      "type": "raw_text",          "weight": 0.10},
            # ShareGPT Multi-Turn Conversational Reasoning (15%)
            {"name": "ShareGPT-MultiTurn",         "path": data_dir / "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", "type": "sharegpt_json",     "weight": 0.15},
            # Sequence Induction & Associative Recall (5%)
            {"name": "Sequence-Patterns",          "path": synthetic_dir / "sequence_patterns.jsonl",                       "type": "raw_text",          "weight": 0.05},
            # Anthropic Helpful & Harmless Multi-Turn Alignment (10%)
            {"name": "Anthropic-RLHF",             "path": data_dir / "anthropic_hh_rlhf.jsonl",                             "type": "conversation_text", "weight": 0.10},
            # UltraChat Multi-Turn Natural Dialogue (8%)
            {"name": "UltraChat-200k",             "path": data_dir / "ultrachat_200k.jsonl",                                "type": "conversation_text", "weight": 0.08},
            # Foundational Teacher Anchor (10%) — Identity, Arithmetic, Logic Ground Truth
            {"name": "PCEI-Teacher",               "path": teacher_curriculum,                                              "type": "prompt_response",   "weight": 0.10},
            # Knowledge Rehearsal STEM Anchor (10%)
            {"name": "FineWeb-Edu",                "path": data_dir / "fineweb_edu.jsonl",                                   "type": "raw_text",          "weight": 0.10},
            # Optional Curricula & Books (Active if available)
            {"name": "OpenHermes-2.5",             "path": data_dir / "openhermes_25.jsonl",                                 "type": "conversation_text", "weight": 0.05},
            {"name": "Cosmopedia",                 "path": philosophy_dir / "cosmopedia_textbooks.txt",                     "type": "text_file",         "weight": 0.02},
            {"name": "TinyStories",                "path": philosophy_dir / "tinystories_bulk.txt",                          "type": "text_file",         "weight": 0.01},
            {"name": "Plato-Republic",             "path": philosophy_dir / "Plato_The_Republic.txt",                        "type": "text_file",         "weight": 0.01},
            {"name": "Infinite-Jest",              "path": modern_dir / "david-foster-wallace-infinite-jest-v2-0.txt",       "type": "text_file",         "weight": 0.01},
        ]

    streamer = BlendedDatasetStreamer(dataset_specs, tokenizer, max_seq_len=args.max_seq_len)

    # Baseline Probe before training
    print("\nRunning initial baseline generation probe...")
    run_probe_generation(nexus, tokenizer, device)

    start_step = checkpoint.get('global_step', 0)
    print(f"\n[Training] Commencing TOKEN-MAXX blended instruction loop (Starting from step {start_step})...")
    accumulated_loss = 0.0
    start_time = time.time()
    t_last_log = time.time()
    tokens_since_log = 0

    optimizer.zero_grad(set_to_none=True)
    current_global_step = start_step

    for step in range(args.steps):
        current_global_step = start_step + step + 1
        if shutdown_requested:
            break

        # Check Credit Guard every 250 steps (strictly halt at $2.00)
        if step > 0 and step % 250 == 0:
            if check_credit_guard():
                print("[CreditGuard] Halting training to preserve cloud credits ($2.00 floor reached).")
                break

        # Calculate Learning Rate with Cosine Decay + Warmup
        if step < args.warmup_steps:
            current_lr = args.base_lr * float(step + 1) / float(max(1, args.warmup_steps))
        else:
            progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
            current_lr = args.min_lr + 0.5 * (args.base_lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Fetch batch from blended stream (supports batch_size > 1 with sequence packing)
        tokens, labels, sample_source = streamer.get_batch(batch_size=args.batch_size, pack=args.pack_sequences)
        tokens = tokens.to(device)
        labels = labels.to(device)

        tokens_since_log += tokens.numel()

        # Integrate clean latent
        with torch.no_grad():
            clean_latent = nexus.core.text_stack.integrate_input(tokens)

        # Autoregressive forward + masked CE loss
        total_loss, ce_val, _, _ = diffusion_training_step(
            nexus, clean_latent, max_T=12,
            ce_target=tokens, ce_weight=1.0,
            labels=labels
        )
        del clean_latent

        # Gradient accumulation backward pass
        loss_scaled = total_loss / args.grad_accum_steps
        loss_scaled.backward()
        accumulated_loss += ce_val
        del total_loss, loss_scaled

        # Step optimizer on accumulation boundary
        if (step + 1) % args.grad_accum_steps == 0:
            nexus.adjust_plasticity(optimizer)
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if device == "mps" and step < args.grad_accum_steps:
                downcast_optimizer_states(optimizer, nexus.system_dtype)

        # Physics tick
        nexus.core.broker.tick_decay()
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
        nexus.ambient_field.tick_consolidation()

        # Telemetry Logging
        if (step + 1) % args.eval_every == 0 or step == 0:
            avg_loss = accumulated_loss / (args.eval_every if step > 0 else 1)
            accumulated_loss = 0.0
            ppl = math.exp(min(avg_loss, 20))
            now = time.time()
            elapsed = now - t_last_log
            tok_s = tokens_since_log / elapsed if elapsed > 0 else 0
            tokens_since_log = 0
            t_last_log = now

            bonds_count = len(nexus.core.broker.active_bonds)
            mean_conf = get_mean_confidence(nexus)

            print(
                f"[Step {current_global_step:06d}/{args.steps+start_step}] Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                f"Tok/s: {tok_s:.1f} | Bonds: {bonds_count} | Conf: {mean_conf:.4f} | "
                f"LR: {current_lr:.2e} | Src: {sample_source}",
                flush=True
            )

        # Live Generation Probe
        if (step + 1) % args.probe_every == 0:
            run_probe_generation(nexus, tokenizer, device)

        # Save Intermediate Checkpoint every save_every steps (e.g. 10k)
        if (step + 1) % args.save_every == 0:
            step_ckpt_path = PROJECT_ROOT / "checkpoints" / f"pcei_750m_reasoning_step_{current_global_step}.pt"
            latest_ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_reasoning_assistant_latest.pt"
            print(f"\n[Checkpoint] Saving intermediate checkpoint to {step_ckpt_path.name} & {latest_ckpt_path.name}...")
            state = {
                'model_state_dict': nexus.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': current_global_step,
                'config': config,
            }
            PCEINexus.save_safe_checkpoint(nexus, state, step_ckpt_path)
            PCEINexus.save_safe_checkpoint(nexus, state, latest_ckpt_path)

            # Write sync signal for local download daemon
            sync_signal_file = PROJECT_ROOT / "checkpoints" / "latest_sync_ready.txt"
            with open(sync_signal_file, "w") as f:
                f.write(f"STEP={current_global_step}\nFILE={step_ckpt_path.name}\nTIME={time.time()}\n")
            print(f"[Sync] Signaled sync ready for step {current_global_step}.")

            # Rolling checkpoint retention: retain only 3 latest step checkpoints to protect disk
            step_ckpts = sorted(
                (PROJECT_ROOT / "checkpoints").glob("pcei_750m_reasoning_step_*.pt"),
                key=lambda p: p.stat().st_mtime
            )
            if len(step_ckpts) > 3:
                for old_ckpt in step_ckpts[:-3]:
                    try:
                        old_ckpt.unlink()
                        print(f"[Prune] Pruned old intermediate checkpoint: {old_ckpt.name}")
                    except Exception as e:
                        pass


    # Save Final Aligned Checkpoint
    final_ckpt_path = PROJECT_ROOT / "checkpoints" / args.output_name
    latest_ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_reasoning_assistant_latest.pt"
    print(f"\n[Final Checkpoint] Saving final aligned model to {final_ckpt_path.name}...")
    state = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': current_global_step,
        'config': config,
    }
    PCEINexus.save_safe_checkpoint(nexus, state, final_ckpt_path)
    PCEINexus.save_safe_checkpoint(nexus, state, latest_ckpt_path)

    sync_signal_file = PROJECT_ROOT / "checkpoints" / "latest_sync_ready.txt"
    with open(sync_signal_file, "w") as f:
        f.write(f"FINAL=True\nFILE={final_ckpt_path.name}\nTIME={time.time()}\n")

    # Final Probe
    print("\nRunning post-training generation probe on final aligned weights...")
    run_probe_generation(nexus, tokenizer, device)

    streamer.close()
    total_elapsed = time.time() - start_time
    print(f"\n================================================================================")
    print(f"  TOKEN-MAXX BLENDED INSTRUCTION TRAINING COMPLETE in {total_elapsed/3600:.2f} hours!")
    print(f"  Final Artifact: checkpoints/{args.output_name}")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
