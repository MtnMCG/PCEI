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

PROMPT_TESTS = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "Who created you?",
    "whatsup",
    "Describe why time flies like an arrow, but fruit flies like a banana?",
    "Tell me a short story about a brave little cat."
]

def run_probes(nexus, tokenizer, device):
    nexus.eval()
    print("\n  ═════════════════ Live Fluency Probes ═════════════════", flush=True)
    with torch.no_grad():
        for q in PROMPT_TESTS:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)

            for _ in range(35):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                next_l = logits[0, -1, :].clone().float()

                # Light repetition penalty on non-syntax tokens
                for tid in set(curr[0, p_len:].tolist()):
                    if tid not in [13, 11, 287, 262, 318, 383]: # Skip ., the, is, The
                        if next_l[tid] > 0: next_l[tid] /= 1.12
                        else: next_l[tid] *= 1.12

                tok_id = torch.argmax(next_l).item()
                if tok_id == 50256: break
                curr = torch.cat([curr, torch.tensor([[tok_id]], device=device)], dim=-1)
                text = tokenizer.decode(curr[0, p_len:].tolist())
                if "\nUSER:" in text or "USER:" in text or "<|endoftext|>" in text: break

            res = tokenizer.decode(curr[0, p_len:].tolist())
            for stop_seq in ["\nUSER:", "USER:", "<|endoftext|>"]:
                if stop_seq in res: res = res.split(stop_seq)[0]
            print(f"    Q: {q}", flush=True)
            print(f"    A: {res.strip()}\n", flush=True)
    print("  ═══════════════════════════════════════════════════════\n", flush=True)
    nexus.train()

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70, flush=True)
    print("  PCEI 750M FINAL FLUENCY ALIGNMENT (CURATED GROUNDED CORPUS)")
    print(f"  Device: {device.upper()} | Steps: 400 | Grad Accum: 4")
    print(f"  Learning Rate: 1.0e-5 -> 2.0e-6 (Gentle Cosine Annealing)")
    print("=" * 70, flush=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_deep_sft_best.pt"
    if not base_ckpt.exists():
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_aligned_sft.pt"

    print(f"[Loading Base] {base_ckpt.name}...", flush=True)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = getattr(tokenizer, "eos_token_id", 50256)

    # Train continuous parameters with strict SFT prompt masking
    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    optimizer = torch.optim.AdamW(continuous_params, lr=1.0e-5, weight_decay=0.01)

    # Load 100% complete, curated dataset
    data_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_fluent_curated_sft.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    random.shuffle(dataset)
    print(f"[Dataset] Loaded {len(dataset):,} pristine, 100% complete SFT pairs.\n", flush=True)

    out_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_fluent_final.pt"

    start_time = time.time()
    best_loss = 999.0

    STEPS = 400
    LR_START = 1.0e-5
    LR_END = 2.0e-6
    GRAD_ACCUM = 4
    MAX_SEQ = 192

    for step in range(1, STEPS + 1):
        # Smooth cosine learning rate decay
        progress = (step - 1) / max(STEPS - 1, 1)
        current_lr = LR_END + 0.5 * (LR_START - LR_END) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            rec = dataset[(step * GRAD_ACCUM + _) % len(dataset)]
            p_text = rec["prompt"].strip()
            r_text = rec["response"].strip()

            full_prompt = f"USER: {p_text}\nASSISTANT:"
            full_response = f" {r_text}<|endoftext|>"

            p_ids = tokenizer.encode(full_prompt, pad=False).tolist()
            r_ids = tokenizer.encode(full_response, pad=False).tolist()

            if len(p_ids) + len(r_ids) > MAX_SEQ:
                r_ids = r_ids[:MAX_SEQ - len(p_ids)]
                if not r_ids or r_ids[-1] != eos_id:
                    r_ids.append(eos_id)

            input_ids = p_ids + r_ids
            # STRICT SFT MASK
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

        torch.nn.utils.clip_grad_norm_(nexus.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if step % 25 == 0 or step == 1:
            elapsed = (time.time() - start_time) / 60.0
            ppl = math.exp(min(accum_loss, 20.0))
            print(f"  [Step {step:4d}/{STEPS}] Response Loss: {accum_loss:.4f} | PPL: {ppl:5.2f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}m", flush=True)

        if step % 50 == 0 or step == STEPS:
            save_payload = {
                "model_state_dict": nexus.state_dict(),
                "config": cfg,
                "step": step,
                "fluent_aligned": True,
                "loss": accum_loss,
                "timestamp": time.time()
            }
            torch.save(save_payload, out_path)
            if accum_loss < best_loss:
                best_loss = accum_loss
                best_path = out_path.with_name(f"{out_path.stem}_best.pt")
                torch.save(save_payload, best_path)

        if step % 100 == 0:
            run_probes(nexus, tokenizer, device)

    print("\n[Complete] PCEI 750M Final Fluency Alignment Complete!", flush=True)
    print(f"Final Model Saved: {out_path}", flush=True)
    run_probes(nexus, tokenizer, device)

if __name__ == "__main__":
    main()
