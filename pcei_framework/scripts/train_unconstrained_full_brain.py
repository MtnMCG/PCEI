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

RAW_PROBES = [
    "What is Rome?",
    "What is Africa?",
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who created you?",
    "whatsup",
    "Why is the sky blue?",
    "Tell me a short story about a brave cat."
]

def run_raw_probes(nexus, tokenizer, device, step):
    nexus.eval()
    print(f"\n  ═════════════════ Real Neural Probes @ Step {step:,} ═════════════════", flush=True)
    with torch.no_grad():
        for q in RAW_PROBES:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)

            # Pure argmax greedy generation directly from logits (0 penalties, 0 masks, 0 bans)
            for _ in range(45):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                tok_id = torch.argmax(logits[0, -1, :]).item()
                if tok_id == 50256: break
                curr = torch.cat([curr, torch.tensor([[tok_id]], device=device)], dim=-1)
                txt = tokenizer.decode(curr[0, p_len:].tolist())
                if "\nUSER:" in txt or "USER:" in txt or "<|endoftext|>" in txt: break

            res = tokenizer.decode(curr[0, p_len:].tolist())
            for stop_seq in ["\nUSER:", "USER:", "<|endoftext|>"]:
                if stop_seq in res: res = res.split(stop_seq)[0]
            print(f"    Q: {q}", flush=True)
            print(f"    A: {res.strip()}\n", flush=True)
    print("  ════════════════════════════════════════════════════════════════════════\n", flush=True)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80, flush=True)
    print("  PCEI 750M FULL BRAIN UNCONSTRAINED TRAINING (NUMERICALLY STABILIZED)")
    print(f"  Device: {device.upper()} | Mode: 100% Raw Weight Updates | Multi-Day Continuous")
    print("  Corpus: 36,125 Structured Pairs (UltraChat + Cosmopedia + Core Entities)")
    print("=" * 80, flush=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_full_brain_best.pt"
    if not base_ckpt.exists():
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_product_best.pt"

    print(f"[Loading Base] Bootstrapping from clean baseline {base_ckpt.name}...", flush=True)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    start_step = ckpt.get("step", 0)
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = 50256

    ternary_weights = []
    continuous_params = []

    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            # Freeze scaling factor and thresholds to strictly prevent activation explosion
            if hasattr(module, 'raw_scale') and module.raw_scale is not None:
                module.raw_scale.requires_grad = False
            if hasattr(module, 'threshold_low') and module.threshold_low is not None:
                module.threshold_low.requires_grad = False
            if hasattr(module, 'threshold_high') and module.threshold_high is not None:
                module.threshold_high.requires_grad = False
            if module.weight.requires_grad:
                ternary_weights.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                continuous_params.append(module.bias)
        else:
            for n_p, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    continuous_params.append(p)

    print(f"[Architecture] Training {len(continuous_params)} Continuous Parameters (Embeddings, Mamba, Norms, KAN).", flush=True)
    print(f"[Architecture] Training {len(ternary_weights)} Ternary Shadow Weights (BitNet {-1, 0, +1} Synapses).", flush=True)

    # Stabilized learning rates with zero weight decay on ternary shadow weights
    optimizer = torch.optim.AdamW([
        {"params": continuous_params, "lr": 1.5e-5, "weight_decay": 0.01},
        {"params": ternary_weights, "lr": 1.5e-5, "weight_decay": 0.0}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=1.0e-6)

    corpus_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_product_grade_corpus.jsonl"
    dataset = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(f"[Dataset] Loaded {len(dataset):,} structured product-grade training pairs.\n", flush=True)

    out_latest = PROJECT_ROOT / "checkpoints" / "pcei_750m_full_brain_latest.pt"
    out_best = PROJECT_ROOT / "checkpoints" / "pcei_750m_full_brain_best.pt"

    start_time = time.time()
    best_loss = ckpt.get("rolling_loss", 4.5)
    rolling_loss = best_loss

    GRAD_ACCUM = 4
    MAX_SEQ = 224

    run_raw_probes(nexus, tokenizer, device, start_step)

    step = start_step
    epoch = ckpt.get("epoch", 0)

    print(f"[Training] Resuming full-brain loop from step {start_step:,}. Running non-stop...", flush=True)

    while True:
        step += 1
        accum_loss = 0.0
        nexus.train()
        step_failed = False

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
            labels = [-100] * len(p_ids) + r_ids

            t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            t_lab = torch.tensor([labels], dtype=torch.long, device=device)

            logits, *_ = nexus.core(text_input=t_in, diffusion_timesteps=1, override_timestep=0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = t_lab[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 12.0:
                step_failed = True
                break

            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            accum_loss += loss.item() / GRAD_ACCUM

        if step_failed:
            print(f"  [Warning] Anomaly or NaN/Inf detected in batch at step {step}. Skipping update.", flush=True)
            optimizer.zero_grad()
            continue

        # Strict NaN-safe gradient clip
        has_bad_grad = False
        for p in nexus.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_bad_grad = True
                    break

        if has_bad_grad:
            print(f"  [Warning] Corrupted gradient detected at step {step}. Skipping update.", flush=True)
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(nexus.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        rolling_loss = 0.95 * rolling_loss + 0.05 * accum_loss if rolling_loss > 0 else accum_loss

        if step % 25 == 0 or step == 1 or step == start_step + 1:
            elapsed = (time.time() - start_time) / 60.0
            ppl = math.exp(min(accum_loss, 20.0))
            roll_ppl = math.exp(min(rolling_loss, 20.0))
            tps = ((step - start_step) * GRAD_ACCUM * MAX_SEQ) / max(time.time() - start_time, 1)
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  [Step {step:6d}] Loss: {accum_loss:.4f} (Avg: {rolling_loss:.4f}) | PPL: {ppl:5.2f} (Avg: {roll_ppl:5.2f}) | LR: {curr_lr:.2e} | Speed: {tps:,.0f} tok/s | Elapsed: {elapsed:.1f}m", flush=True)

        if step % 250 == 0:
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

            run_raw_probes(nexus, tokenizer, device, step)

if __name__ == "__main__":
    main()
