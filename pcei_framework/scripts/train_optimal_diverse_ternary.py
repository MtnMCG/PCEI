#!/usr/bin/env python3
"""
pcei_framework/scripts/train_optimal_diverse_ternary.py

PCEI 100M Optimal High-Diversity True BitNet 1.58b Training Pipeline.
Balances multi-domain streams in real-time to avoid kindergarten mode collapse:
  - 35% Advanced STEM & Textbooks (Cosmopedia, Wikipedia)
  - 30% High-Order Chain-of-Thought & Reasoning (Anthropic Claude <think>, Exemplars)
  - 20% Philosophy & Classical/Modern Literature (Plato, Aristotle, Descartes, Nietzsche, Books)
  - 15% Foundational Fluency & Concept Grounding (Grammar, Congruent Concepts)

Features:
  - Memory-efficient continuous streaming (no 1GB file RAM crashes)
  - Strict True BitNet 1.58b ternary quantization {-1.0, 0.0, +1.0}
  - Accelerated via Horner's polynomial evaluation in FastKANMixer
  - Continuous sequence packing for 100% compute utilization on Apple Silicon MPS
  - Atomic checkpointing & periodic generation probes
  - Graceful exit on Ctrl+C (SIGINT)
"""

import sys
import os
import time
import math
import signal
import json
import random
import argparse
from pathlib import Path
from typing import List, Generator

# Add project root
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


def extract_text(record: dict) -> str:
    """Extracts raw dialogue or text from various jsonl schemas."""
    # 1. Claude / Direct format
    if "text" in record and record["text"]:
        return str(record["text"]).strip()
    
    # 2. Prompt / Response
    prompt = record.get("prompt")
    response = record.get("response")
    if prompt or response:
        p = str(prompt).strip() if prompt else ""
        r = str(response).strip() if response else ""
        if p and r:
            return f"USER: {p}\nASSISTANT: {r}"
        return p or r
        
    # 3. Instruction / Output (Alpaca / ShareGPT)
    inst = record.get("instruction")
    out = record.get("output")
    inp = record.get("input", "")
    if inst or out:
        parts = []
        if inst:
            parts.append(f"USER: {str(inst).strip()}")
        if inp:
            parts.append(f"Context: {str(inp).strip()}")
        if out:
            parts.append(f"ASSISTANT: {str(out).strip()}")
        return "\n".join(parts)
        
    # 4. Messages schema
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = "USER" if m.get("role", "") in ["user", "human"] else "ASSISTANT"
                content = str(m.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
            
    # 5. Conversations schema
    convs = record.get("conversations")
    if isinstance(convs, list) and convs:
        parts = []
        for turn in convs:
            if isinstance(turn, dict):
                speaker = "USER" if turn.get("from", "") in ["human", "user"] else "ASSISTANT"
                val = str(turn.get("value", "")).strip()
                if val:
                    parts.append(f"{speaker}: {val}")
        if parts:
            return "\n".join(parts)
            
    return ""


class MemoryEfficientStream:
    """Streams documents one by one from a list of files without loading all into memory."""
    def __init__(self, file_paths: List[Path], chunk_chars: int = 2000):
        self.file_paths = [p for p in file_paths if p.exists()]
        self.chunk_chars = chunk_chars
        self.file_idx = 0
        random.shuffle(self.file_paths)

    def doc_generator(self) -> Generator[str, None, None]:
        if not self.file_paths:
            while True:
                yield "The Physical Continuum Emergent Intelligence operates on physical principles."

        while True:
            path = self.file_paths[self.file_idx]
            self.file_idx = (self.file_idx + 1) % len(self.file_paths)
            if self.file_idx == 0:
                random.shuffle(self.file_paths)

            try:
                if path.suffix == ".jsonl":
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    rec = json.loads(line)
                                    text = extract_text(rec)
                                    if text and len(text) > 15:
                                        yield text
                                except Exception:
                                    continue
                elif path.suffix in [".txt", ".md"]:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        buf = ""
                        for line in f:
                            buf += line
                            if len(buf) >= self.chunk_chars:
                                yield buf.strip()
                                buf = ""
                        if buf.strip():
                            yield buf.strip()
                elif path.suffix == ".parquet":
                    import pyarrow.parquet as pq
                    try:
                        pfile = pq.ParquetFile(path)
                        for batch in pfile.iter_batches(batch_size=512):
                            cols = batch.schema.names
                            if "title" in cols and "body" in cols:
                                # Reddit corpus
                                titles = batch.column("title").to_pylist()
                                bodies = batch.column("body").to_pylist()
                                for t, b in zip(titles, bodies):
                                    t_str = str(t or "").strip()
                                    b_str = str(b or "").strip()
                                    if b_str and len(b_str) > 20:
                                        if t_str and t_str not in b_str:
                                            yield f"{t_str}\n\n{b_str}"
                                        else:
                                            yield b_str
                            elif "instruction" in cols and "response" in cols:
                                # Dolly 15k
                                insts = batch.column("instruction").to_pylist()
                                resps = batch.column("response").to_pylist()
                                ctxs = batch.column("context").to_pylist() if "context" in cols else [None] * len(insts)
                                for inst, resp, ctx in zip(insts, resps, ctxs):
                                    inst_str = str(inst or "").strip()
                                    resp_str = str(resp or "").strip()
                                    ctx_str = str(ctx or "").strip()
                                    if inst_str and resp_str:
                                        if ctx_str:
                                            yield f"USER: {inst_str}\nContext: {ctx_str}\nASSISTANT: {resp_str}"
                                        else:
                                            yield f"USER: {inst_str}\nASSISTANT: {resp_str}"
                            elif "text" in cols and "role" in cols:
                                # OASST1
                                texts = batch.column("text").to_pylist()
                                roles = batch.column("role").to_pylist()
                                langs = batch.column("lang").to_pylist() if "lang" in cols else ["en"] * len(texts)
                                for text, role, lang in zip(texts, roles, langs):
                                    if (lang == "en" or not lang) and text:
                                        r = "USER" if role in ["prompter", "user"] else "ASSISTANT"
                                        yield f"{r}: {str(text).strip()}"
                            elif "text" in cols:
                                for text in batch.column("text").to_pylist():
                                    t = str(text or "").strip()
                                    if len(t) > 20:
                                        yield t
                    except Exception:
                        continue
            except Exception:
                continue


class DiverseMultiStreamPacker:
    """
    Packs continuous token sequences drawing from 5 stratified domains:
      - Stream 1 (30%): Natural Human Dialogue & Reddit Corpus (3.18 GB Reddit + Dolly 15k + OASST1 + UltraChat + SmolTalk)
      - Stream 2 (25%): High-Order Reasoning & Claude CoT
      - Stream 3 (20%): Advanced STEM & Textbooks (Cosmopedia)
      - Stream 4 (15%): Philosophy & Classic/Modern Literature
      - Stream 5 (10%): Syntactic Grounding & Foundation Concepts (TinyStories capped at 10%)
    """
    def __init__(self, tokenizer: PCEITokenizer, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer = []

        data_dir = PROJECT_ROOT / "data"

        # 1. Natural Human Dialogue & Reddit Corpus (30%)
        conversational_files = [
            data_dir / "conversational" / "dolly_15k.parquet",
            data_dir / "conversational" / "oasst1_train.parquet",
            data_dir / "conversational" / "oasst1_val.parquet",
            data_dir / "synthetic" / "ultrachat_conversations.jsonl",
            data_dir / "synthetic" / "smoltalk_dialogues.jsonl",
        ]
        reddit_files = sorted(list((data_dir / "conversational" / "reddit").glob("*.parquet")))
        conversational_files.extend(reddit_files)

        # 2. Reasoning & CoT (15% - personality grounding disabled to prevent sycophancy)
        reasoning_files = [
            data_dir / "synthetic" / "anthropic_claude.jsonl",
            data_dir / "high_quality" / "sharegpt_logic.jsonl",
            data_dir / "synthetic" / "beast_reasoning.jsonl",
            data_dir / "synthetic" / "instruction_tuning.jsonl",
            # Disabled personality/identity grounding to eliminate memorization loops:
            # data_dir / "curriculum" / "pcei_foundation_teacher.jsonl",
            # data_dir / "curriculum" / "council_teacher.jsonl"
        ]

        # 3. STEM & Textbooks (12%)
        stem_files = [
            data_dir / "raw" / "text" / "mass_reasoning" / "cosmopedia_textbooks.txt",
            data_dir / "wikitext_valid.txt"
        ]

        # 4. Philosophy & Modern Literature (8%)
        phil_files = [
            data_dir / "raw" / "text" / "mass_reasoning" / "Plato_The_Republic.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Aristotle_Nicomachean_Ethics.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Aristotle_Politics.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Descartes_Discourse_On_Method.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Nietzsche_Beyond_Good_And_Evil.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Nietzsche_Thus_Spake_Zarathustra.txt",
            data_dir / "raw" / "text" / "mass_reasoning" / "Sun_Tzu_The_Art_Of_War.txt"
        ]
        phil_files.extend(list((data_dir / "raw" / "text" / "modern_books").glob("*.txt")))
        phil_files.extend(list((data_dir / "raw" / "text" / "mass_reasoning").glob("book_*.txt")))

        # 5. Syntactic Grounding & Foundation Concepts (5%)
        grounding_files = [
            data_dir / "raw" / "text" / "mass_reasoning" / "tinystories_bulk.txt",
            data_dir / "synthetic" / "congruent_concepts.jsonl"
        ]

        self.streams = [
            ("Conversational/Reddit", MemoryEfficientStream(conversational_files), 0.60),
            ("Claude CoT/Reasoning", MemoryEfficientStream(reasoning_files), 0.15),
            ("STEM/Textbooks", MemoryEfficientStream(stem_files), 0.12),
            ("Philosophy/Lit", MemoryEfficientStream(phil_files), 0.08),
            ("Syntactic/Grounding", MemoryEfficientStream(grounding_files), 0.05)
        ]
        self.stream_iters = [s[1].doc_generator() for s in self.streams]
        self.weights = [s[2] for s in self.streams]

    def _refill_buffer(self, min_tokens: int):
        while len(self.buffer) < min_tokens:
            # Pick a domain stream according to distribution
            stream_idx = random.choices(range(len(self.streams)), weights=self.weights, k=1)[0]
            doc = next(self.stream_iters[stream_idx])
            if doc and doc.strip():
                tokens = self.tokenizer.encode_raw(doc[:4000])
                if tokens:
                    if tokens[-1] != self.tokenizer.eos_id:
                        tokens.append(self.tokenizer.eos_id)
                    self.buffer.extend(tokens)

    def get_batch(self, batch_size: int, device: str = "cpu") -> dict:
        total_tokens = batch_size * self.max_seq_len
        self._refill_buffer(total_tokens)

        batch_slice = self.buffer[:total_tokens]
        self.buffer = self.buffer[total_tokens:]

        tokens = torch.tensor(batch_slice, dtype=torch.long).view(batch_size, self.max_seq_len).to(device)
        return {"text": tokens}


def probe_generation(nexus: PCEINexus, tokenizer: PCEITokenizer, device: torch.device):
    """Runs quick live generation probes to observe fluency, reasoning, and vocabulary in real-time."""
    nexus.eval()
    test_prompts = [
        "USER: Explain why ice floats on water and why this is critical for aquatic ecosystems.\nASSISTANT:",
        "USER: If all roses are flowers and some flowers fade quickly, what can we deduce?\nASSISTANT:",
        "USER: What is the relationship between potential energy and kinetic energy in an oscillation?\nASSISTANT:"
    ]
    print("\n" + "─" * 60)
    print("  LIVE KNOWLEDGE & REASONING PROBE")
    print("─" * 60)

    for p in test_prompts:
        ids = tokenizer.encode(p).unsqueeze(0).to(device)
        curr = ids.clone()
        generated = []
        max_new = 40

        with torch.no_grad():
            for _ in range(max_new):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                next_logits = logits[0, -1, :].clone().float()
                # Mask pad_id (0) and eos_id (50256)
                next_logits[0] = -1e4
                next_logits[50256] = -1e4

                # Additive repetition penalty on recent tokens
                if generated:
                    for tok in set(generated[-20:]):
                        next_logits[tok] -= 2.5

                # Top-k (k=15) temperature sampling
                topk_vals, topk_idx = torch.topk(next_logits, k=15)
                probs = F.softmax(topk_vals / 0.5, dim=-1)
                sel = torch.multinomial(probs, 1).item()
                next_tok = topk_idx[sel].item()
                generated.append(next_tok)
                curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
                if curr.shape[1] > 256:
                    curr = curr[:, -256:]

        text_out = tokenizer.decode(torch.tensor(generated))
        print(f"\n[Prompt] {p.splitlines()[0]}", flush=True)
        print(f"[Model]  {text_out.strip()}", flush=True)
    print("─" * 60 + "\n", flush=True)
    nexus.train()


def main():
    parser = argparse.ArgumentParser(description="Optimal High-Diversity Ternary Training for PCEI 100M")
    parser.add_argument("--steps", type=int, default=5000, help="Number of training steps to run (default: 5,000)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length (default: 128)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (default: 2.5e-5)")
    parser.add_argument("--save-every", type=int, default=250, help="Checkpoint interval (default: 250 steps)")
    parser.add_argument("--probe-every", type=int, default=500, help="Generation probe interval (default: 500 steps)")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset optimizer momentum states and re-initialize LR schedule")
    args = parser.parse_args()

    print("=" * 70)
    print("  PCEI 100M — OPTIMAL HIGH-DIVERSITY TRUE TERNARY TRAINING")
    print(f"  Domain Allocation: 60% Conversational | 15% Reasoning | 12% STEM | 8% Lit | 5% Grounding")
    print(f"  Target Steps: {args.steps:,} | Batch: {args.batch_size} | SeqLen: {args.seq_len}")
    print(f"  Quantization: Strict True BitNet {{-1.0, 0.0, +1.0}}")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    # Checkpoint resolution: prefer latest diverse ternary checkpoint, fallback to 100m.pt, then true_ternary.pt
    ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m_diverse_ternary.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m_true_ternary.pt"
    if not ckpt_path.exists():
        print(f"Error: No checkpoint found in checkpoints directory.")
        sys.exit(1)

    print(f"[Checkpoint] Loading weights from: {ckpt_path.name}")
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

    # Audit initial weights
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
    packer = DiverseMultiStreamPacker(tokenizer=tokenizer, max_seq_len=args.seq_len)
    print(f"[Data] Multi-Stream Stratified Packer initialized successfully.")

    # Optimizer & Scheduler
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)
    if "optimizer_state_dict" in ckpt and not args.reset_optimizer:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Optimizer] Restored optimizer state from checkpoint.")
        except Exception:
            print("[Optimizer] Initialized fresh optimizer state.")
    else:
        print("[Optimizer] Reset fresh optimizer state (cleared momentum/history).")

    # Explicitly enforce target LR and attach Cosine Annealing to eliminate ternary dithering
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        param_group['initial_lr'] = args.lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25000, eta_min=5e-6)
    print(f"[Scheduler] Active: CosineAnnealingLR (base_lr={args.lr:.2e} -> eta_min=5e-6)")

    start_step = ckpt.get("global_step", 0)
    current_step = start_step
    target_end_step = (start_step + args.steps) if args.steps > 0 else float('inf')
    start_time = time.time()
    should_exit = False

    def handle_signal(sig, frame):
        nonlocal should_exit
        print("\n[Exit Signal] Shutdown requested. Calcifying weights atomically before exit...", flush=True)
        should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n" + "─" * 70)
    mode_str = f"Steps {start_step:,} → {target_end_step:,}" if args.steps > 0 else f"CONTINUOUS DAEMON MODE (from Step {start_step:,})"
    print(f"  TRAINING ACTIVE: {mode_str}")
    print("─" * 70, flush=True)

    step_loss_history = []
    log_every = 25
    out_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_100m_diverse_ternary.pt"
    telemetry_file = PROJECT_ROOT / "logs" / "training_telemetry.jsonl"
    telemetry_file.parent.mkdir(parents=True, exist_ok=True)

    while current_step < target_end_step and not should_exit:
        current_step += 1
        t_step_start = time.perf_counter()
        optimizer.zero_grad()

        # Stream batch
        batch = packer.get_batch(args.batch_size, device=device)
        tokens = batch["text"]

        # Latent Integration
        with torch.no_grad():
            clean_latent = nexus.core.text_stack.integrate_input(tokens)

        # Diffusion Step
        total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
            nexus, clean_latent, max_T=3,
            ce_target=tokens, ce_weight=0.3,
            use_min_snr=False
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Ambient Consolidation Tick
        nexus.ambient_field.tick_consolidation()

        loss_val = total_loss.item()
        step_loss_history.append(loss_val)

        step_duration = time.perf_counter() - t_step_start

        # Logging
        if current_step % log_every == 0:
            avg_loss = sum(step_loss_history[-log_every:]) / log_every
            ppl = math.exp(min(20, avg_loss))
            tok_per_sec = (args.batch_size * args.seq_len) / max(step_duration, 1e-4)
            elapsed = time.time() - start_time
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {current_step:,} | Loss: {avg_loss:.4f} | PPL: {ppl:8.2f} | "
                  f"LR: {curr_lr:.2e} | Speed: {tok_per_sec:6.1f} tok/s | Elapsed: {elapsed/60:.1f}m", flush=True)

            try:
                with open(telemetry_file, "a") as tf:
                    tf.write(json.dumps({
                        "step": current_step,
                        "loss": round(avg_loss, 4),
                        "ppl": round(ppl, 2),
                        "lr": round(curr_lr, 7),
                        "tok_per_sec": round(tok_per_sec, 1),
                        "elapsed_seconds": round(elapsed, 1),
                        "timestamp": time.time()
                    }) + "\n")
            except Exception:
                pass

        # Periodic Live Probing
        if current_step % args.probe_every == 0:
            probe_generation(nexus, tokenizer, device)

        # Periodic Atomic Checkpointing
        if current_step % args.save_every == 0:
            save_dict = {
                "model_state_dict": nexus.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": current_step,
                "config": config,
                "ternary_spec": "BitNet_1.58b_strict_three_state",
                "training_domain": "multi_stream_stratified_diversity"
            }
            PCEINexus.save_safe_checkpoint(nexus, save_dict, out_ckpt, create_backup=False)
            PCEINexus.save_safe_checkpoint(nexus, save_dict, PROJECT_ROOT / "checkpoints" / "pcei_100m.pt", create_backup=False)

        if "mps" in str(device) and current_step % 25 == 0:
            torch.mps.empty_cache()

    # Final Save
    total_time = time.time() - start_time
    print(f"\n[Session Finished] Completed in {total_time/60:.2f} minutes.")
    print(f"[Session Finished] Final Global Step: {current_step:,} (+{current_step - start_step:,} new steps).")

    final_save_dict = {
        "model_state_dict": nexus.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": current_step,
        "config": config,
        "ternary_spec": "BitNet_1.58b_strict_three_state",
        "training_domain": "multi_stream_stratified_diversity",
        "training_duration_seconds": total_time
    }
    PCEINexus.save_safe_checkpoint(nexus, final_save_dict, out_ckpt, create_backup=True)
    PCEINexus.save_safe_checkpoint(nexus, final_save_dict, PROJECT_ROOT / "checkpoints" / "pcei_100m.pt", create_backup=False)
    print(f"[Saved] Checkpoint written to: {out_ckpt.name} and pcei_100m.pt")


if __name__ == "__main__":
    main()
