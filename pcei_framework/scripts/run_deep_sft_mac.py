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

def clean_tag(text: str) -> str:
    text = text.strip()
    for tag in ["USER:", "ASSISTANT:", "SYSTEM:"]:
        if text.startswith(tag):
            text = text[len(tag):].strip()
    for tag in ["\nUSER:", "\nASSISTANT:", "ASSISTANT:", "USER:"]:
        if text.endswith(tag):
            text = text[:-len(tag)].strip()
    return text.strip()

def build_balanced_sft_dataset(data_dir: Path):
    samples = []

    # 1. Teacher Conversational & Identity (Clean, 3819 items)
    p_teacher = data_dir / "curriculum" / "pcei_clean_teacher_conversational.jsonl"
    if p_teacher.exists():
        with open(p_teacher, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r: samples.append((p, r, "dialogue"))

    # 2. Pure Conversational Clean (3378 items)
    p_pure = data_dir / "curriculum" / "pcei_pure_conversational_clean.jsonl"
    if p_pure.exists():
        with open(p_pure, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r: samples.append((p, r, "conversational"))

    # 3. Instruction Tuning (sample 4000)
    p_inst = data_dir / "synthetic" / "instruction_tuning.jsonl"
    if p_inst.exists():
        with open(p_inst, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
            random.shuffle(lines)
            for line in lines[:4000]:
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", "") or rec.get("instruction", ""))
                r = clean_tag(rec.get("response", "") or rec.get("output", ""))
                if p and r: samples.append((p, r, "instruction"))

    # 4. Claude CoT Reasoning (sample 2500)
    p_cot = data_dir / "synthetic" / "anthropic_claude_unified_cot.jsonl"
    if p_cot.exists():
        with open(p_cot, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
            random.shuffle(lines)
            for line in lines[:2500]:
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r: samples.append((p, r, "reasoning"))

    # 5. Arithmetic (sample 1200 - keep arithmetic sharp without overwhelming dialogue)
    p_arith = data_dir / "curriculum" / "kindergarten_arithmetic.jsonl"
    if p_arith.exists():
        with open(p_arith, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
            random.shuffle(lines)
            for line in lines[:1200]:
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r: samples.append((p, r, "arithmetic"))

    random.shuffle(samples)
    print(f"[Dataset] Balanced SFT Dataset Assembled: {len(samples):,} high-quality instruction pairs.", flush=True)
    return samples

PROMPT_PROBES = [
    "What is the capital of France?",
    "Who created you?",
    "whatsup",
    "What is 2 + 2?",
    "Tell me a short story about a brave little cat."
]

def run_probes(nexus, tokenizer, device):
    nexus.eval()
    print("\n  --- Live Generative Probes ---", flush=True)
    with torch.no_grad():
        for q in PROMPT_PROBES:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)

            for _ in range(45):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                next_l = logits[0, -1, :].clone().float()

                # Repetition penalty
                for tid in set(curr[0, p_len:].tolist()):
                    if next_l[tid] > 0: next_l[tid] /= 1.15
                    else: next_l[tid] *= 1.15

                probs = F.softmax(next_l / 0.35, dim=-1)
                tok_id = torch.multinomial(probs, 1)
                if tok_id.item() == 50256: break
                curr = torch.cat([curr, tok_id.unsqueeze(0)], dim=-1)
                text = tokenizer.decode(curr[0, p_len:].tolist())
                if "\nUSER:" in text or "USER:" in text or "<|endoftext|>" in text: break

            res = tokenizer.decode(curr[0, p_len:].tolist())
            for stop_seq in ["\nUSER:", "USER:", "<|endoftext|>"]:
                if stop_seq in res: res = res.split(stop_seq)[0]
            print(f"    Q: {q}", flush=True)
            print(f"    A: {res.strip()}\n", flush=True)
    nexus.train()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcei_750m_aligned_sft.pt")
    parser.add_argument("--output", type=str, default="checkpoints/pcei_750m_deep_sft.pt")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr-start", type=float, default=2.5e-5)
    parser.add_argument("--lr-end", type=float, default=4.0e-6)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=200)
    args = parser.parse_args()

    device = args.device
    print("=" * 70, flush=True)
    print("  PCEI 750M DEEP LOCAL SFT TRAINING RUN (MAC APPLE SILICON)")
    print(f"  Device: {device.upper()} | Steps: {args.steps:,} | Grad Accum: {args.grad_accum}")
    print(f"  Learning Rate: {args.lr_start} -> {args.lr_end} (Cosine Annealing)")
    print("=" * 70, flush=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        # Fallback to 6h_best
        ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_6h_best.pt"

    print(f"[Loading Checkpoint] {ckpt_path.name}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = getattr(tokenizer, "eos_token_id", 50256)

    # Freeze ternary weights during pure conversational SFT; train continuous parameters
    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    total_params_trained = sum(p.numel() for p in continuous_params)
    print(f"[Architecture] Training {total_params_trained:,} continuous parameters via SFT.", flush=True)

    optimizer = torch.optim.AdamW(continuous_params, lr=args.lr_start, weight_decay=0.01)

    dataset = build_balanced_sft_dataset(PROJECT_ROOT / "data")
    data_idx = 0

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    start_time = time.time()
    best_loss = 999.0

    print(f"\n[Training] Starting 2,000-Step Deep SFT Alignment...", flush=True)

    for step in range(1, args.steps + 1):
        # Cosine learning rate schedule
        progress = (step - 1) / max(args.steps - 1, 1)
        current_lr = args.lr_end + 0.5 * (args.lr_start - args.lr_end) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        accum_loss = 0.0

        for _ in range(args.grad_accum):
            p_text, r_text, domain = dataset[data_idx % len(dataset)]
            data_idx += 1

            full_prompt = f"USER: {p_text}\nASSISTANT:"
            full_response = f" {r_text}<|endoftext|>"

            p_ids = tokenizer.encode(full_prompt, pad=False).tolist()
            r_ids = tokenizer.encode(full_response, pad=False).tolist()

            if len(p_ids) + len(r_ids) > args.max_seq_len:
                r_ids = r_ids[:args.max_seq_len - len(p_ids)]
                if not r_ids or r_ids[-1] != eos_id:
                    r_ids.append(eos_id)

            input_ids = p_ids + r_ids
            # STRICT SFT MASK: Prompt tokens masked with -100
            labels = [-100] * len(p_ids) + r_ids

            t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            t_lab = torch.tensor([labels], dtype=torch.long, device=device)

            logits, *_ = nexus.core(text_input=t_in, diffusion_timesteps=1, override_timestep=0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = t_lab[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(nexus.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Prevent MPS cache buildup
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if step % args.log_every == 0 or step == 1:
            elapsed = (time.time() - start_time) / 60.0
            ppl = math.exp(min(accum_loss, 20.0))
            print(f"  [Step {step:5d}/{args.steps}] Loss: {accum_loss:.4f} | PPL: {ppl:5.2f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}m", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            save_payload = {
                "model_state_dict": nexus.state_dict(),
                "config": cfg,
                "step": step,
                "sft_loss": accum_loss,
                "timestamp": time.time()
            }
            torch.save(save_payload, out_path)
            if accum_loss < best_loss:
                best_loss = accum_loss
                best_path = out_path.with_name(f"{out_path.stem}_best.pt")
                torch.save(save_payload, best_path)

        if step % args.eval_every == 0:
            run_probes(nexus, tokenizer, device)

    print("\n[Complete] Deep Local SFT Alignment finished successfully!", flush=True)
    print(f"Final Model saved to: {out_path}", flush=True)

if __name__ == "__main__":
    main()
