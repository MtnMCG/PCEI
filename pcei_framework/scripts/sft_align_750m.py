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

def load_sft_dataset(data_dir: Path):
    samples = []
    
    # 1. Clean teacher conversational (3,800+ clean dialogs)
    teacher_path = data_dir / "curriculum" / "pcei_clean_teacher_conversational.jsonl"
    if teacher_path.exists():
        with open(teacher_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r:
                    samples.append((p, r))
                    
    # 2. Arithmetic (5,000 diverse arithmetic examples)
    arith_path = data_dir / "curriculum" / "kindergarten_arithmetic.jsonl"
    if arith_path.exists():
        with open(arith_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 5000: break
                if not line.strip(): continue
                rec = json.loads(line)
                p = clean_tag(rec.get("prompt", ""))
                r = clean_tag(rec.get("response", ""))
                if p and r:
                    samples.append((p, r))

    random.shuffle(samples)
    print(f"[Dataset] Loaded {len(samples)} clean SFT instruction-response pairs.")
    return samples

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcei_750m_6h_best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/pcei_750m_aligned_sft.pt")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    device = args.device
    print("=" * 65)
    print("  PCEI 750M Clean SFT Alignment (Strict Prompt Masking)")
    print(f"  Device: {device.upper()} | Steps: {args.steps} | LR: {args.lr}")
    print("=" * 65)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    print(f"[Loading] {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = getattr(tokenizer, "eos_token_id", 50256)

    # Train continuous parameters for fast alignment without destabilizing ternary
    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    optimizer = torch.optim.AdamW(continuous_params, lr=args.lr, weight_decay=0.01)

    dataset = load_sft_dataset(PROJECT_ROOT / "data")
    data_idx = 0

    print(f"\n[Alignment] Training for {args.steps} steps with STRICT prompt masking...")
    nexus.train()
    optimizer.zero_grad()

    start_t = time.time()
    for step in range(1, args.steps + 1):
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            p_text, r_text = dataset[data_idx % len(dataset)]
            data_idx += 1

            full_prompt = f"USER: {p_text}\nASSISTANT: "
            full_response = f"{r_text}<|endoftext|>"

            p_ids = tokenizer.encode(full_prompt, pad=False).tolist()
            r_ids = tokenizer.encode(full_response, pad=False).tolist()

            if len(p_ids) + len(r_ids) > 256:
                r_ids = r_ids[:256 - len(p_ids)]
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

        # Prevent Metal Performance Shaders (MPS) cache buildup
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if step % 25 == 0 or step == 1:
            elapsed = time.time() - start_t
            ppl = math.exp(min(accum_loss, 20.0))
            print(f"  [Step {step:4d}/{args.steps}] SFT Response Loss: {accum_loss:.4f} | PPL: {ppl:5.2f} | Time: {elapsed:.1f}s", flush=True)
            
            # Save progress checkpoint
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
            save_payload = {
                "model_state_dict": nexus.state_dict(),
                "config": cfg,
                "step": ckpt.get("step", 65000) + step,
                "sft_aligned": True,
                "timestamp": time.time()
            }
            torch.save(save_payload, out_path)

    # Save aligned checkpoint
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    
    save_payload = {
        "model_state_dict": nexus.state_dict(),
        "config": cfg,
        "step": ckpt.get("step", 65000) + args.steps,
        "sft_aligned": True,
        "timestamp": time.time()
    }
    torch.save(save_payload, out_path)
    print(f"\n[Saved] Aligned checkpoint saved to: {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
