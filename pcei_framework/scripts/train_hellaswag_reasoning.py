#!/usr/bin/env python3
"""
pcei_framework/scripts/train_hellaswag_reasoning.py

PCEI 100M Dual-Anchor Generative & Contrastive Commonsense Reasoning Optimizer.
Aligns PCEI 100M BitNet Mamba on HellaSwag narrative causality to achieve >= 50% accuracy:
  - Generative Causal LM loss on true continuation (prevents logit collapse)
  - Contrastive Margin Ranking Loss enforcing s_true >= s_j + margin
  - Contrastive Choice Cross-Entropy over candidate log-probabilities
  - Physics-aware parameter group optimization with BitNet unfreezing LR
  - Periodic evaluation against official HellaSwag validation slice
  - Milestone checkpointing at checkpoints/pcei_100m_hellaswag_50pct.pt
"""

import sys
import os
import time
import math
import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer
from scripts.eval_hellaswag import evaluate_hellaswag

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "curriculum" / "hellaswag_train.jsonl"
VAL_PATH = DATA_DIR / "benchmarks" / "hellaswag_val_slice.jsonl"


def load_hellaswag_records(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def main():
    parser = argparse.ArgumentParser(description="PCEI 100M Dual-Anchor HellaSwag Optimizer")
    parser.add_argument("--base-checkpoint", type=str, default="pcei_100m_hellaswag_best.pt")
    parser.add_argument("--steps", type=int, default=6000, help="Total micro-steps (samples)")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Peak learning rate for BitNet plasticity")
    parser.add_argument("--min-lr", type=float, default=3.0e-6, help="Floor learning rate")
    parser.add_argument("--tau", type=float, default=0.25, help="Softmax temperature for choice logits")
    parser.add_argument("--margin", type=float, default=0.35, help="Margin ranking gap threshold")
    parser.add_argument("--eval-every", type=int, default=50, help="Validation frequency in optimizer steps")
    parser.add_argument("--eval-samples", type=int, default=100, help="Validation sample count for quick checks")
    parser.add_argument("--target-acc", type=float, default=0.50, help="Target accuracy threshold")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_ckpt_path = CHECKPOINT_DIR / args.base_checkpoint
    if not base_ckpt_path.exists():
        base_ckpt_path = CHECKPOINT_DIR / "pcei_100m_conversational.pt"
    if not base_ckpt_path.exists():
        base_ckpt_path = CHECKPOINT_DIR / "pcei_100m.pt"

    output_ckpt_path = CHECKPOINT_DIR / "pcei_100m_hellaswag_best.pt"
    goal_ckpt_path = CHECKPOINT_DIR / "pcei_100m_hellaswag_50pct.pt"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "train_hellaswag.log"

    banner = f"""
{"═" * 75}
  PCEI 100M DUAL-ANCHOR HELLASWAG REASONING OPTIMIZER (PHASE 5)
  Target Device: {device.upper()} | Base Checkpoint: {base_ckpt_path.name}
  Target Goal: >= {args.target_acc * 100:.1f}% HellaSwag Accuracy
  LR: {args.lr:.2e} -> {args.min_lr:.2e} | Margin: {args.margin} | Tau: {args.tau} | Accum: {args.accum_steps}
{"═" * 75}
"""
    print(banner, flush=True)
    with open(log_file, "a") as f:
        f.write(banner)

    tokenizer = PCEITokenizer()
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(base_ckpt_path)
    config = PCEIConfig(layers_text=l_t, layers_img=l_i, layers_audio=l_a, d_model_text=d_t, d_model_img=d_i, d_model_audio=d_a, vocab_size=50257)
    nexus = PCEINexus(config=config, device=device)
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    nexus.to(device)
    nexus.train()

    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=1e-4)

    train_data = load_hellaswag_records(TRAIN_PATH)
    val_data = load_hellaswag_records(VAL_PATH)
    print(f"[Data] Loaded {len(train_data):,} training samples, {len(val_data):,} validation samples.")

    # Cosine scheduler calibrated to total optimizer steps
    total_opt_steps = args.steps // args.accum_steps
    def lr_lambda(opt_step: int) -> float:
        progress = opt_step / max(total_opt_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = args.min_lr / args.lr
        return ratio + (1.0 - ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_acc = ckpt.get("val_acc", 0.35)
    rolling_clm_losses = []
    rolling_choice_losses = []
    rolling_margin_losses = []
    train_idx = 0
    random.shuffle(train_data)
    t_start = time.time()

    print(f"[Training] Commencing {args.steps} samples ({total_opt_steps} optimizer steps) with Hard-Negative Mining (Current Best: {best_val_acc*100:.2f}%)...\n", flush=True)

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for step in range(1, args.steps + 1):
        if train_idx >= len(train_data):
            train_idx = 0
            random.shuffle(train_data)

        item = train_data[train_idx]
        train_idx += 1

        ctx = item.get("ctx", "")
        if not ctx and "ctx_a" in item:
            ctx = item["ctx_a"] + " " + item.get("ctx_b", "").capitalize()
        endings = item.get("endings", [])
        if len(endings) != 4 or "label" not in item:
            continue

        label = int(item["label"])
        ctx_tokens = tokenizer.encode(ctx).tolist()
        ctx_len = len(ctx_tokens)

        option_scores = []
        clm_loss = None

        for opt_idx, opt in enumerate(endings):
            full_text = ctx + " " + opt
            full_ids = tokenizer.encode(full_text).unsqueeze(0).to(device)
            if full_ids.size(1) > 512:
                full_ids = full_ids[:, :512]

            logits, *_ = nexus.core(
                text_input=full_ids,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state
            )

            shift_logits = logits[:, :-1, :]
            shift_labels = full_ids[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            start_tok = max(0, min(ctx_len - 1, gathered.size(1) - 1))
            ending_log_probs = gathered[0, start_tok:]
            
            score = ending_log_probs.mean() if ending_log_probs.numel() > 0 else torch.tensor(-100.0, device=device)
            option_scores.append(score)

            if opt_idx == label:
                clm_loss = -score

        scores_tensor = torch.stack(option_scores)
        target_tensor = torch.tensor(label, device=device)

        # 1. Calibrated Softmax choice loss
        choice_loss = F.cross_entropy(scores_tensor.unsqueeze(0) / args.tau, target_tensor.unsqueeze(0))

        # 2. Smooth Tri-Distractor Margin ranking loss (proven non-oscillating in Phase 3)
        true_score = scores_tensor[label]
        margin_penalties = [F.relu(args.margin - (true_score - scores_tensor[j])) for j in range(4) if j != label]
        margin_loss = sum(margin_penalties) / 3.0

        # 3. Dual-Anchor Combined Loss: CLM Anchor + Smooth Margin + Softmax Choice
        total_loss = 1.0 * (clm_loss if clm_loss is not None else 0.0) + 0.5 * margin_loss + 0.5 * choice_loss

        scaled_loss = total_loss / args.accum_steps
        scaled_loss.backward()
        accum_count += 1

        rolling_clm_losses.append(clm_loss.item() if clm_loss is not None else 0.0)
        rolling_choice_losses.append(choice_loss.item())
        rolling_margin_losses.append(margin_loss.item())
        if len(rolling_choice_losses) > 50:
            rolling_clm_losses.pop(0)
            rolling_choice_losses.pop(0)
            rolling_margin_losses.pop(0)

        # Clear spikes & broker buffers
        if hasattr(nexus.core, "cached_spikes") and isinstance(nexus.core.cached_spikes, dict):
            nexus.core.cached_spikes.clear()
        if hasattr(nexus.core, "broker") and hasattr(nexus.core.broker, "active_bonds"):
            nexus.core.broker.active_bonds.clear()

        if accum_count >= args.accum_steps:
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            # Logging every 25 optimizer steps
            opt_step = step // args.accum_steps
            if opt_step > 0 and opt_step % 25 == 0:
                avg_clm = sum(rolling_clm_losses) / len(rolling_clm_losses)
                avg_choice = sum(rolling_choice_losses) / len(rolling_choice_losses)
                avg_margin = sum(rolling_margin_losses) / len(rolling_margin_losses)
                elapsed_min = (time.time() - t_start) / 60.0
                cur_lr = optimizer.param_groups[0]['lr']
                log_line = f"  Step {opt_step:>4d} | CLM: {avg_clm:.3f} | Choice: {avg_choice:.3f} | Margin: {avg_margin:.3f} | LR: {cur_lr:.2e} | Elapsed: {elapsed_min:.1f}m"
                print(log_line, flush=True)
                with open(log_file, "a") as f:
                    f.write(log_line + "\n")

            # Periodic Validation
            if opt_step > 0 and opt_step % args.eval_every == 0:
                print(f"\n[Validation] Running HellaSwag evaluation at Step {opt_step}...", flush=True)
                nexus.eval()
                val_samples = val_data[:args.eval_samples]
                val_acc = evaluate_hellaswag(nexus, tokenizer, val_samples, device=device)
                nexus.train()

                val_log = f"\n  >>> Milestone Step {opt_step} | HellaSwag Quick Val ({len(val_samples)}): {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%) <<<\n"
                print(val_log, flush=True)
                with open(log_file, "a") as f:
                    f.write(val_log + "\n")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({'model_state_dict': nexus.state_dict(), 'val_acc': val_acc, 'step': opt_step}, output_ckpt_path)
                    print(f"  [Checkpoint Updated] New best checkpoint saved to {output_ckpt_path.name} ({val_acc*100:.2f}%)", flush=True)

                # Check if target threshold met
                if val_acc >= args.target_acc:
                    print(f"\n{'*' * 75}\n  TARGET REACHED: {val_acc*100:.2f}% >= {args.target_acc*100:.2f}%!\n  Running Full 500-Sample Validation Audit...\n{'*' * 75}\n", flush=True)
                    nexus.eval()
                    full_acc = evaluate_hellaswag(nexus, tokenizer, val_data[:500], device=device)
                    nexus.train()

                    full_log = f"\n  >>> FULL BENCHMARK AUDIT (500 samples): {full_acc*100:.2f}% <<<\n"
                    print(full_log, flush=True)
                    with open(log_file, "a") as f:
                        f.write(full_log + "\n")

                    if full_acc >= args.target_acc:
                        torch.save({'model_state_dict': nexus.state_dict(), 'val_acc': full_acc, 'step': opt_step}, goal_ckpt_path)
                        print(f"\n{'!' * 75}\n  GOAL FULFILLED! Saved milestone checkpoint to {goal_ckpt_path.name} ({full_acc*100:.2f}%)\n{'!' * 75}\n", flush=True)
                        return

    print(f"\n[Run Concluded] Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
