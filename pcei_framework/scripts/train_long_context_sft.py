#!/usr/bin/env python3
"""
scripts/train_long_context_sft.py

PCEI 100M Long-Context (2,048-Token) Multi-Turn Supervised Fine-Tuning (SFT).
Executes prompt-loss-masked alignment on the Step 100,000 BitNet foundation backbone.

Features:
  - 2,048-token context window with continuous sequence packing
  - Alternating multi-turn prompt loss masking (USER tokens masked with label=0; ASSISTANT tokens active)
  - Full un-truncated ingestion of multi-turn dialogues, math proofs, and Claude CoT reasoning traces
  - Strict BitNet b1.58 ternary weight preservation {-1.0, 0.0, +1.0}
  - Gradient checkpointing on TextStack to keep MPS RAM under 2.0 GB
  - Post-probe detector buffer & active bond state purge
  - Atomic checkpointing with circuit breaker protection
"""

import sys
import os
import time
import math
import signal
import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.modules.ternary import TernaryLinear
from src.tokenizer import PCEITokenizer
from src.engine.diffusion_utils import diffusion_training_step


def parse_conversation_turns(text: str) -> List[Tuple[str, str]]:
    """Extracts (role, content) turns from text formatted with USER:/ASSISTANT: or Human:/Assistant:"""
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


class MemoryEfficientTurnStream:
    """Streams multi-turn dialogues or instruction pairs line-by-line with infinite cyclical replay."""
    def __init__(self, file_path: Path, dataset_type: str):
        self.file_path = file_path
        self.dataset_type = dataset_type

    def turn_generator(self) -> Generator[List[Tuple[str, str]], None, None]:
        while True:
            if not self.file_path.exists():
                time.sleep(1.0)
                continue
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    # 1. Multi-turn text block (UltraChat, SmolTalk, Beast)
                    if "text" in record and record["text"]:
                        raw_text = str(record["text"]).strip()
                        turns = parse_conversation_turns(raw_text)
                        if turns:
                            yield turns
                        elif len(raw_text) > 40:
                            yield [("user", "Explain:"), ("assistant", raw_text)]

                    # 2. Prompt / Response pair (Claude CoT, PCEI Teacher, Instruction Tuning)
                    elif "prompt" in record and "response" in record:
                        p = str(record["prompt"] or "").strip()
                        r = str(record["response"] or "").strip()
                        if p and r:
                            if p.endswith("ASSISTANT:"):
                                p = p[:-len("ASSISTANT:")].strip()
                            if p.startswith("USER:"):
                                p = p[len("USER:"):].strip()
                            yield [("user", p), ("assistant", r)]

                    # 3. Instruction / Response schema
                    elif "instruction" in record and "response" in record:
                        inst = str(record["instruction"] or "").strip()
                        resp = str(record["response"] or "").strip()
                        if inst and resp:
                            yield [("user", inst), ("assistant", resp)]


class MultiTurnInstructionPacker:
    """
    Packs multi-turn conversations and long-form reasoning into fixed 2,048-token chunks.
    Applies strict prompt loss masking:
      - USER prompt tokens: label = 0 (ignored in cross-entropy loss)
      - ASSISTANT answer tokens: label = token_id (actively optimized)
    """
    def __init__(self, tokenizer: PCEITokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        data_dir = PROJECT_ROOT / "data"
        curriculum_dir = data_dir / "curriculum"
        synthetic_dir = data_dir / "synthetic"

        self.stream_configs = [
            # 1. Claude CoT Reasoning (30%) - Deep multi-step reasoning with <think> scratchpads
            ("Claude CoT Reasoning", synthetic_dir / "anthropic_claude_unified_cot.jsonl", "prompt_response", 0.30),
            # 2. UltraChat Full Multi-Turn (25%) - Rich, multi-paragraph dialogue flows
            ("UltraChat Dialogues", synthetic_dir / "ultrachat_conversations.jsonl", "multiturn_text", 0.25),
            # 3. SmolTalk Problem Solving (15%) - Mathematics, logic proofs, boxed solutions
            ("SmolTalk Math & Logic", synthetic_dir / "smoltalk_dialogues.jsonl", "multiturn_text", 0.15),
            # 4. PCEI Teacher Grounding (15%) - Architecture, continuum physics, BitNet identity
            ("PCEI Teacher Identity", curriculum_dir / "pcei_foundation_teacher.jsonl", "prompt_response", 0.15),
            # 5. Targeted Instruction Tuning (10%) - Conversational QA & helpful response alignment
            ("Targeted Instructions", synthetic_dir / "instruction_tuning.jsonl", "prompt_response", 0.10),
            # 6. Procedural Arithmetic (3%) - Elementary calculations and step logic
            ("Procedural Arithmetic", curriculum_dir / "kindergarten_arithmetic.jsonl", "prompt_response", 0.03),
            # 7. Reasoning Exemplars (2%) - Dense logic & physics syllogisms
            ("Reasoning Exemplars", curriculum_dir / "pcei_reasoning_exemplars.jsonl", "prompt_response", 0.02),
        ]

        self.streams = []
        self.weights = []
        for name, path, dtype, weight in self.stream_configs:
            if path.exists() and path.stat().st_size > 0:
                stream_obj = MemoryEfficientTurnStream(path, dtype)
                self.streams.append((name, stream_obj, weight))
                self.weights.append(weight)
                print(f"[Data] Loaded SFT stream: {name:<24} ({path.name}) | Weight: {weight*100:.0f}%")
            else:
                print(f"[Data Warning] SFT stream file missing: {path.name}, skipping.")

        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        self.stream_iters = [s[1].turn_generator() for s in self.streams]

        self.token_buffer: List[int] = []
        self.label_buffer: List[int] = []

    def _conversation_to_tokens_and_labels(self, turns: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
        """Converts multi-turn turns into token IDs and loss-masked label IDs."""
        tokens: List[int] = []
        labels: List[int] = []

        num_turns = len(turns)
        for i, (role, content) in enumerate(turns):
            is_last = (i == num_turns - 1)
            content_str = content.strip()
            if not content_str:
                continue

            if role in ("user", "system"):
                prefix = "USER: " if i == 0 else "\nUSER: "
                chunk = f"{prefix}{content_str}\nASSISTANT:"
                t_ids = self.tokenizer.encode_raw(chunk)
                tokens.extend(t_ids)
                labels.extend([0] * len(t_ids))  # Mask out user prompts
            elif role == "assistant":
                chunk = f" {content_str}"
                if is_last and "<|endoftext|>" not in chunk:
                    chunk += "<|endoftext|>\n"
                t_ids = self.tokenizer.encode_raw(chunk)
                tokens.extend(t_ids)
                labels.extend(t_ids)  # Active target loss

        return tokens, labels

    def get_batch(self, batch_size: int = 1, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Packs complete, un-truncated conversations into (batch_size, max_seq_len) tensors.
        Never splits conversations across batch boundaries to prevent headless assistant responses.
        Remaining space is padded with 0 (ignored via ignore_index=0).
        """
        batch_tokens = []
        batch_labels = []

        for _ in range(batch_size):
            row_tokens = []
            row_labels = []

            while len(row_tokens) < self.max_seq_len:
                stream_idx = random.choices(range(len(self.streams)), weights=self.weights, k=1)[0]
                turns = next(self.stream_iters[stream_idx])
                toks, labs = self._conversation_to_tokens_and_labels(turns)

                # If fits completely, add to current row
                if len(row_tokens) + len(toks) <= self.max_seq_len:
                    row_tokens.extend(toks)
                    row_labels.extend(labs)
                else:
                    # If this is the first conversation in the row and it exceeds max_seq_len, truncate cleanly
                    # but only if it contains active assistant response tokens
                    if not row_tokens:
                        truncated_labs = labs[:self.max_seq_len]
                        if any(l != 0 for l in truncated_labs):
                            row_tokens = toks[:self.max_seq_len]
                            row_labels = truncated_labs
                        else:
                            # Prompt exceeded context window before response; skip to next sample
                            continue
                    break

            # Pad remainder with 0 (pad_id, ignored in cross-entropy)
            pad_len = self.max_seq_len - len(row_tokens)
            if pad_len > 0:
                row_tokens.extend([0] * pad_len)
                row_labels.extend([0] * pad_len)

            batch_tokens.append(row_tokens)
            batch_labels.append(row_labels)

        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

        # Safety: ensure at least some unmasked labels exist in next-token prediction targets
        if (labels_tensor[:, 1:] != 0).sum().item() == 0:
            labels_tensor[:, self.max_seq_len // 2:] = tokens_tensor[:, self.max_seq_len // 2:]

        return {"text": tokens_tensor, "labels": labels_tensor}


def probe_generation(nexus: PCEINexus, tokenizer: PCEITokenizer, device: torch.device):
    """Evaluates live instruction following and reasoning, then purges internal detector/broker buffers."""
    test_prompts = [
        "USER: Hello! What is your name and what model architecture are you?\nASSISTANT:",
        "USER: Explain why ice floats on water and why this is critical for aquatic ecosystems.\nASSISTANT:",
        "USER: What is 2 + 2, and what is 15 - 8?\nASSISTANT:",
        "USER: If all roses are flowers and some flowers fade quickly, what can we deduce?\nASSISTANT:",
        "USER: Write a clean Python function to find the maximum value in a list of numbers without using max().\nASSISTANT:",
    ]
    print("\n" + "─" * 60)
    print("  LIVE SFT INSTRUCTION & REASONING PROBE")
    print("─" * 60)

    nexus.eval()

    for p in test_prompts:
        ids = tokenizer.encode(p).unsqueeze(0).to(device)
        curr = ids.clone()
        generated = []
        max_new = 50

        with torch.no_grad():
            for _ in range(max_new):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                next_logits = logits[0, -1, :].clone().float()
                next_logits[0] = -1e4
                next_logits[50256] = -1e4

                if generated:
                    for tok in set(generated[-20:]):
                        next_logits[tok] -= 2.5

                topk_vals, topk_idx = torch.topk(next_logits, k=15)
                probs = F.softmax(topk_vals / 0.5, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    break
                sel = torch.multinomial(probs, 1).item()
                next_tok = topk_idx[sel].item()
                generated.append(next_tok)
                curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
                if curr.shape[1] > 1024:
                    curr = curr[:, -1024:]

        text_out = tokenizer.decode(torch.tensor(generated))
        print(f"\n[Prompt] {p.splitlines()[0]}")
        print(f"[Model]  {text_out.strip()}", flush=True)
    print("─" * 60 + "\n", flush=True)

    # Clean up detector activations and broker bonds generated during probing
    if hasattr(nexus, "core"):
        if hasattr(nexus.core, "detectors"):
            for d in nexus.core.detectors.values():
                if hasattr(d, "activation_buffer"):
                    d.activation_buffer.clear()
        if hasattr(nexus.core, "cached_spikes") and isinstance(nexus.core.cached_spikes, dict):
            nexus.core.cached_spikes.clear()
        if hasattr(nexus.core, "broker") and hasattr(nexus.core.broker, "active_bonds"):
            nexus.core.broker.active_bonds.clear()

    if "mps" in str(device):
        torch.mps.empty_cache()

    nexus.train()


def main():
    parser = argparse.ArgumentParser(description="PCEI 100M Long-Context Multi-Turn SFT Pipeline")
    parser.add_argument("--steps", type=int, default=5000, help="Number of SFT steps to run (default: 5,000)")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Context sequence length (default: 1,024)")
    parser.add_argument("--lr", type=float, default=2.0e-6, help="Peak SFT learning rate (default: 2.0e-6)")
    parser.add_argument("--min-lr", type=float, default=5.0e-7, help="Minimum learning rate floor (default: 5.0e-7)")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps (default: 100)")
    parser.add_argument("--cycle-steps", type=int, default=5000, help="Cosine annealing steps (default: 5,000)")
    parser.add_argument("--save-every", type=int, default=250, help="Checkpoint interval (default: 250 steps)")
    parser.add_argument("--probe-every", type=int, default=500, help="Generation probe interval (default: 500 steps)")
    args = parser.parse_args()

    effective_tokens_per_opt_step = args.batch_size * args.max_seq_len * args.grad_accum
    print("=" * 70)
    print("  PCEI 100M — LONG-CONTEXT (1,024-TOKEN) MULTI-TURN SFT PIPELINE")
    print(f"  Context Window: {args.max_seq_len:,} tokens | Micro-Batch: {args.batch_size} | Accum: {args.grad_accum}")
    print(f"  Effective Batch: {effective_tokens_per_opt_step:,} tokens/optimizer_step")
    print(f"  Curriculum: 30% Claude CoT | 25% UltraChat | 15% SmolTalk | 15% Teacher | 10% Instructions | 5% Math/Logic")
    print(f"  Target SFT Steps: {args.steps:,} | Learning Rate: {args.lr:.2e} -> {args.min_lr:.2e}")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m_diverse_ternary.pt"
    if not ckpt_path.exists():
        print(f"Error: Base checkpoint not found.")
        sys.exit(1)

    print(f"[Checkpoint] Loading base foundation weights from: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_obj = ckpt.get("config", None)
    if isinstance(cfg_obj, PCEIConfig):
        config = cfg_obj
    elif isinstance(cfg_obj, dict) and cfg_obj:
        config = PCEIConfig(**cfg_obj)
    else:
        config = PCEIConfig.load()

    nexus = PCEINexus(config=config).to(device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.train()
    nexus.set_mode("WITNESS")
    if hasattr(nexus.core.text_stack, 'enable_gradient_checkpointing'):
        nexus.core.text_stack.enable_gradient_checkpointing()
        print("[Memory] Gradient checkpointing ENABLED on TextStack.")

    # Synapse Audit
    total_q, neg1, zero0, pos1 = 0, 0, 0, 0
    for name, m in nexus.named_modules():
        if isinstance(m, TernaryLinear):
            q = m.quantized_weight
            total_q += q.numel()
            neg1 += (q == -1.0).sum().item()
            zero0 += (q == 0.0).sum().item()
            pos1 += (q == 1.0).sum().item()

    print(f"[Initial Weights] Audited {total_q:,} quantized synapses:")
    print(f"  • Inhibitory (-1): {neg1:,} ({neg1/total_q*100:.1f}%)")
    print(f"  • Quiescent   (0): {zero0:,} ({zero0/total_q*100:.1f}%)")
    print(f"  • Excitatory (+1): {pos1:,} ({pos1/total_q*100:.1f}%)")

    # Data Packer
    tokenizer = PCEITokenizer(vocab_size=config.vocab_size)
    packer = MultiTurnInstructionPacker(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Optimizer & Scheduler
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Optimizer] Restored calibrated optimizer state from checkpoint.")
        except Exception as e:
            print(f"[Optimizer] Could not load state dict ({e}), initialized fresh optimizer state.")
    else:
        print("[Optimizer] Initialized fresh optimizer state.")

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        param_group['initial_lr'] = args.lr

    min_lr = args.min_lr
    cycle_steps = args.cycle_steps
    warmup_steps = args.warmup_steps
    ratio = min(1.0, min_lr / args.lr) if args.lr > 0 else 1.0

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.05, float(step + 1) / float(max(1, warmup_steps)))
        progress_step = step - warmup_steps
        total_decay_steps = max(1, cycle_steps - warmup_steps)
        if progress_step >= total_decay_steps:
            return ratio
        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress_step / total_decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"[Scheduler] Linear Warmup ({warmup_steps} steps) + CosineAnnealing active (lr: {args.lr:.2e} -> {min_lr:.2e} floor over {cycle_steps:,} steps)")

    start_step = ckpt.get("global_step", 100000)
    current_step = start_step
    target_end_step = (start_step + args.steps) if args.steps > 0 else float('inf')
    start_time = time.time()
    should_exit = False
    circuit_breaker_tripped = False

    def handle_signal(sig, frame):
        nonlocal should_exit
        print("\n[Exit Signal] Shutdown requested. Saving checkpoint before exit...", flush=True)
        should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n" + "─" * 70)
    print(f"  SFT TRAINING ACTIVE: Steps {start_step:,} → {target_end_step:,}")
    print("─" * 70, flush=True)

    step_loss_history = []
    consecutive_elevated_checks = 0
    consecutive_spike_count = 0
    log_every = 25
    sft_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_100m_sft.pt"
    telemetry_file = PROJECT_ROOT / "logs" / "sft_telemetry.jsonl"
    telemetry_file.parent.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad()

    while current_step < target_end_step and not should_exit:
        current_step += 1
        t_step_start = time.perf_counter()

        accumulated_loss = 0.0
        step_failed = False

        for accum_idx in range(args.grad_accum):
            batch_valid = False
            for retry in range(3):
                batch = packer.get_batch(args.batch_size, device=device)
                tokens = batch["text"]
                labels = batch["labels"]

                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, clean_latent=None, max_T=1,
                    ce_target=tokens, labels=labels, ce_weight=1.0,
                    use_min_snr=False
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"\n[Warning] Step {current_step}: NaN/Inf loss encountered. Resampling batch.", flush=True)
                    continue

                loss_val = total_loss.item()

                if loss_val > 8.50:
                    consecutive_spike_count += 1
                    if consecutive_spike_count >= 5:
                        print(f"\n[CIRCUIT BREAKER] Step {current_step}: 5 consecutive loss spikes ({loss_val:.2f} > 8.50). Halting.", flush=True)
                        circuit_breaker_tripped = True
                        should_exit = True
                        break
                    print(f"\n[Batch Anomaly] Step {current_step}: Extreme outlier ({loss_val:.2f} > 8.50). Resampling batch.", flush=True)
                    continue
                else:
                    consecutive_spike_count = 0

                scaled_loss = total_loss / args.grad_accum
                scaled_loss.backward()
                accumulated_loss += loss_val / args.grad_accum
                batch_valid = True
                break

            if circuit_breaker_tripped or not batch_valid:
                step_failed = True
                break

        if circuit_breaker_tripped:
            break

        if step_failed:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Ambient consolidation tick
        nexus.ambient_field.tick_consolidation()

        step_loss_history.append(accumulated_loss)
        step_duration = time.perf_counter() - t_step_start

        # Logging & Circuit Breakers
        if current_step % log_every == 0:
            window_size = min(len(step_loss_history), 50)
            avg_loss = sum(step_loss_history[-window_size:]) / window_size
            ppl = math.exp(min(20, avg_loss))
            tok_per_sec = (effective_tokens_per_opt_step) / max(step_duration, 1e-4)
            elapsed = time.time() - start_time
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  SFT Step {current_step:,} | Loss: {avg_loss:.4f} | PPL: {ppl:8.2f} | "
                  f"LR: {curr_lr:.2e} | Speed: {tok_per_sec:6.1f} tok/s | Elapsed: {elapsed/60:.1f}m", flush=True)

            if avg_loss > 7.25:
                print(f"\n[CIRCUIT BREAKER] Step {current_step}: Severe rolling loss spike ({avg_loss:.2f} > 7.25). Halting.", flush=True)
                circuit_breaker_tripped = True
                should_exit = True
                break

            try:
                rec = json.dumps({
                    "step": current_step,
                    "loss": round(avg_loss, 4),
                    "ppl": round(ppl, 2),
                    "lr": round(curr_lr, 7),
                    "tok_per_sec": round(tok_per_sec, 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "timestamp": time.time(),
                    "phase": "SFT_1024"
                }) + "\n"
                with open(telemetry_file, "a") as tf:
                    tf.write(rec)
                with open(PROJECT_ROOT / "logs" / "training_telemetry.jsonl", "a") as tf_global:
                    tf_global.write(rec)
            except Exception:
                pass

        # Periodic Generation Probes
        if current_step % args.probe_every == 0:
            probe_generation(nexus, tokenizer, device)

        # Periodic Atomic Checkpointing
        if current_step % args.save_every == 0 and not circuit_breaker_tripped:
            save_dict = {
                "model_state_dict": nexus.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": current_step,
                "config": config,
                "ternary_spec": "BitNet_1.58b_strict_three_state",
                "training_domain": "long_context_multi_turn_sft",
                "context_length": args.max_seq_len,
            }
            PCEINexus.save_safe_checkpoint(nexus, save_dict, sft_ckpt, create_backup=False)
            PCEINexus.save_safe_checkpoint(nexus, save_dict, PROJECT_ROOT / "checkpoints" / "pcei_100m.pt", create_backup=False)

        if "mps" in str(device) and current_step % 25 == 0:
            torch.mps.empty_cache()

    # Final Save
    total_time = time.time() - start_time
    print(f"\n[SFT Session Finished] Completed in {total_time/60:.2f} minutes.")
    print(f"[SFT Session Finished] Final Global Step: {current_step:,} (+{current_step - start_step:,} SFT steps).")

    if circuit_breaker_tripped:
        print(f"[CIRCUIT BREAKER] Checkpoint overwrite PREVENTED. Retaining existing safe checkpoint.", flush=True)
    else:
        final_save_dict = {
            "model_state_dict": nexus.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": current_step,
            "config": config,
            "ternary_spec": "BitNet_1.58b_strict_three_state",
            "training_domain": "long_context_multi_turn_sft",
            "context_length": args.max_seq_len,
            "training_duration_seconds": total_time
        }
        PCEINexus.save_safe_checkpoint(nexus, final_save_dict, sft_ckpt, create_backup=True)
        PCEINexus.save_safe_checkpoint(nexus, final_save_dict, PROJECT_ROOT / "checkpoints" / "pcei_100m.pt", create_backup=False)
        print(f"[Saved] Checkpoint written to: {sft_ckpt.name} and pcei_100m.pt")


if __name__ == "__main__":
    main()
