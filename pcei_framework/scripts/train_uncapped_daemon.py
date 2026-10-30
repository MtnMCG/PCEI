import sys
import os
import time
import math
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pcei import PCEINexus
from src.modules.ternary import TernaryLinear
from src.tokenizer import PCEITokenizer

PRODUCT_PROBES = [
    "What is Rome?",
    "What is Africa?",
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who created you?",
    "whatsup",
    "Why is the sky blue?",
    "Tell me a short story about a brave cat."
]

def run_probes(nexus, tokenizer, device, step):
    nexus.eval()
    print(f"\n  ═════════════════ Product Probes @ Step {step:,} ═════════════════", flush=True)
    with torch.no_grad():
        for q in PRODUCT_PROBES:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)
            gen_ids = []

            for _ in range(50):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                logits_last = logits[0, -1, :].clone().float()

                # Clean n-gram repetition blocking (prevents loops without penalizing syntax)
                if len(gen_ids) >= 2:
                    recent = tuple(gen_ids[-2:])
                    banned = []
                    for i in range(len(gen_ids) - 2):
                        if (gen_ids[i], gen_ids[i+1]) == recent:
                            banned.append(gen_ids[i+2])
                    for b in banned:
                        logits_last[b] = -1e4

                tok_id = torch.argmax(logits_last).item()
                if tok_id == 50256: break
                gen_ids.append(tok_id)
                curr = torch.cat([curr, torch.tensor([[tok_id]], device=device)], dim=-1)
                text = tokenizer.decode(curr[0, p_len:].tolist())
                if "\nUSER:" in text or "USER:" in text or "<|endoftext|>" in text: break

            res = tokenizer.decode(curr[0, p_len:].tolist())
            for stop_seq in ["\nUSER:", "USER:", "<|endoftext|>"]:
                if stop_seq in res: res = res.split(stop_seq)[0]
            print(f"    Q: {q}", flush=True)
            print(f"    A: {res.strip()}\n", flush=True)
    print("  ═══════════════════════════════════════════════════════════════════\n", flush=True)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80, flush=True)
    print("  PCEI 750M UNCAPPED PRODUCT-GRADE TRAINING ENGINE (MULTI-DAY DAEMON)")
    print(f"  Device: {device.upper()} | Target Steps: 100,000+ (Continuous Infinite Loop)")
    print("  Corpus: 36,125 Structured Pairs (UltraChat + Cosmopedia + Core Grounding)")
    print("=" * 80, flush=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_general_sft.pt"
    if not base_ckpt.exists():
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_continuous_latest.pt"

    print(f"[Loading Base] Bootstrapping from {base_ckpt.name}...", flush=True)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = 50256

    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    print(f"[Architecture] Preserving {len(ternary_set)} ternary synaptic matrices; training {len(continuous_params)} continuous parameters.", flush=True)

    # Steady enterprise learning rate with gentle decay floor
    LR_BASE = 2.0e-5
    LR_MIN = 3.0e-6
    DECAY_STEPS = 50000
    optimizer = torch.optim.AdamW(continuous_params, lr=LR_BASE, weight_decay=0.01)

    corpus_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_product_grade_corpus.jsonl"
    dataset = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(f"[Dataset] Loaded {len(dataset):,} structured product-grade pairs.\n", flush=True)

    out_latest = PROJECT_ROOT / "checkpoints" / "pcei_750m_product_latest.pt"
    out_best = PROJECT_ROOT / "checkpoints" / "pcei_750m_product_best.pt"

    start_time = time.time()
    best_loss = 999.0
    rolling_loss = 0.0

    GRAD_ACCUM = 4
    MAX_SEQ = 224

    # Run quick initial probe check
    run_probes(nexus, tokenizer, device, 0)

    step = 0
    epoch = 0

    print("[Daemon] Entering continuous uncapped loop. Running 24/7...", flush=True)

    while True:
        step += 1
        # Dynamic cosine learning rate schedule with floor
        progress = min(step / DECAY_STEPS, 1.0)
        current_lr = LR_MIN + 0.5 * (LR_BASE - LR_MIN) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        accum_loss = 0.0
        nexus.train()

        for _ in range(GRAD_ACCUM):
            data_idx = (step * GRAD_ACCUM + _) % len(dataset)
            if data_idx == 0:
                epoch += 1

            rec = dataset[data_idx]
            p_text = rec["prompt"].strip()
            r_text = rec["response"].strip()

            full_prompt = f"USER: {p_text}\nASSISTANT:"
            full_response = f" {r_text}<|endoftext|>"

            p_ids = tokenizer.encode(full_prompt).tolist()
            r_ids = tokenizer.encode(full_response).tolist()

            if len(p_ids) + len(r_ids) > MAX_SEQ:
                r_ids = r_ids[:MAX_SEQ - len(p_ids)]
                if not r_ids or r_ids[-1] != eos_id:
                    r_ids.append(eos_id)

            input_ids = p_ids + r_ids
            # Strict prompt masking: loss calculated strictly on response
            labels = [-100] * len(p_ids) + r_ids

            t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            t_lab = torch.tensor([labels], dtype=torch.long, device=device)

            logits, *_ = nexus.core(text_input=t_in, diffusion_timesteps=1, override_timestep=0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = t_lab[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        rolling_loss = 0.95 * rolling_loss + 0.05 * accum_loss if rolling_loss > 0 else accum_loss

        if step % 25 == 0 or step == 1:
            elapsed = (time.time() - start_time) / 60.0
            ppl = math.exp(min(accum_loss, 20.0))
            roll_ppl = math.exp(min(rolling_loss, 20.0))
            tps = (step * GRAD_ACCUM * MAX_SEQ) / max(time.time() - start_time, 1)
            print(f"  [Step {step:6d}] Loss: {accum_loss:.4f} (Avg: {rolling_loss:.4f}) | PPL: {ppl:5.2f} (Avg: {roll_ppl:5.2f}) | LR: {current_lr:.2e} | Speed: {tps:,.0f} tok/s | Elapsed: {elapsed:.1f}m", flush=True)

        if step % 500 == 0:
            save_payload = {
                "step": step,
                "epoch": epoch,
                "model_state_dict": nexus.state_dict(),
                "config": cfg,
                "loss": accum_loss,
                "rolling_loss": rolling_loss,
                "timestamp": time.time()
            }
            torch.save(save_payload, out_latest)
            print(f"  [Checkpoint] Periodic save written to {out_latest.name} (Step {step:,})", flush=True)

            if rolling_loss < best_loss:
                best_loss = rolling_loss
                torch.save(save_payload, out_best)
                print(f"  [*] New Best Checkpoint saved to {out_best.name} (Rolling Loss: {best_loss:.4f})", flush=True)

            run_probes(nexus, tokenizer, device, step)

if __name__ == "__main__":
    main()
