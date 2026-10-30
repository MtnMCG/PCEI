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

def stream_text_file(path: Path, max_chars: int = 1500):
    while True:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                buf = ""
                for line in f:
                    buf += line
                    if len(buf) >= max_chars:
                        yield buf.strip()
                        buf = ""
                if buf.strip():
                    yield buf.strip()
        except Exception:
            yield "The universe operates under fundamental physical laws."

def stream_jsonl_file(path: Path):
    while True:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                        p = rec.get("prompt") or rec.get("instruction") or ""
                        r = rec.get("response") or rec.get("output") or ""
                        if p and r:
                            yield f"USER: {p}\nASSISTANT: {r}"
                        elif r:
                            yield str(r)
                    except:
                        pass
        except Exception:
            yield "Hello! How can I assist you today?"

DAEMON_PROBES = [
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
    print(f"\n  ═════════════════ Continuous Daemon Probes @ Step {step:,} ═════════════════", flush=True)
    with torch.no_grad():
        for q in DAEMON_PROBES:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)

            for _ in range(50):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                next_l = logits[0, -1, :].clone().float()

                # Light non-syntax penalty
                for tid in set(curr[0, p_len:].tolist()):
                    if tid not in [13, 11, 287, 262, 318, 383]:
                        if next_l[tid] > 0: next_l[tid] /= 1.15
                        else: next_l[tid] *= 1.15

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
    print("  ═══════════════════════════════════════════════════════════════════════════\n", flush=True)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 75, flush=True)
    print("  PCEI 750M CONTINUOUS PRODUCT-GRADE TRAINING DAEMON")
    print(f"  Device: {device.upper()} | Unconstrained Step Loop | Auto-Saving Every 500 Steps")
    print("  Corpus: 1.1 GB Cosmopedia Textbooks + 20k UltraChat + TinyStories Bulk")
    print("=" * 75, flush=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_general_sft.pt"
    if not base_ckpt.exists():
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_6h_best.pt"

    print(f"[Daemon] Bootstrapping from {base_ckpt.name}...", flush=True)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()
    eos_id = 50256

    # Unfreeze continuous layers (embeddings, Mamba projections, norm layers, KAN splines)
    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    print(f"[Architecture] Preserving {len(ternary_set)} ternary synaptic matrices; training {len(continuous_params)} continuous parameters.", flush=True)

    # Steady enterprise learning rate
    optimizer = torch.optim.AdamW(continuous_params, lr=3.0e-5, weight_decay=0.01)

    # Data streams: Cosmopedia (knowledge), UltraChat (dialogue), TinyStories (flow)
    cosmo_path = PROJECT_ROOT / "data" / "raw" / "text" / "mass_reasoning" / "cosmopedia_textbooks.txt"
    ultrachat_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_general_sft_20k.jsonl"
    stories_path = PROJECT_ROOT / "data" / "raw" / "text" / "mass_reasoning" / "tinystories_bulk.txt"

    streams = [
        stream_text_file(cosmo_path),
        stream_jsonl_file(ultrachat_path),
        stream_text_file(stories_path)
    ]

    out_latest = PROJECT_ROOT / "checkpoints" / "pcei_750m_continuous_latest.pt"
    out_best = PROJECT_ROOT / "checkpoints" / "pcei_750m_continuous_best.pt"

    start_time = time.time()
    best_loss = 999.0
    step = 0
    GRAD_ACCUM = 4
    MAX_SEQ = 256

    print("\n[Daemon] Continuous training loop active. Running unconstrained...", flush=True)

    while True:
        step += 1
        accum_loss = 0.0
        nexus.train()

        for _ in range(GRAD_ACCUM):
            # Interleave streams: 50% knowledge, 35% dialogue, 15% narrative
            stream_picker = random.random()
            if stream_picker < 0.50:
                text_chunk = next(streams[0])
            elif stream_picker < 0.85:
                text_chunk = next(streams[1])
            else:
                text_chunk = next(streams[2])

            tokens = tokenizer.encode(text_chunk).tolist()
            if not tokens or len(tokens) < 8:
                continue

            if len(tokens) > MAX_SEQ:
                start_idx = random.randint(0, len(tokens) - MAX_SEQ)
                tokens = tokens[start_idx : start_idx + MAX_SEQ]

            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            logits, *_ = nexus.core(text_input=inp, diffusion_timesteps=1, override_timestep=0)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if step % 25 == 0 or step == 1:
            elapsed = (time.time() - start_time) / 60.0
            ppl = math.exp(min(accum_loss, 20.0))
            tps = (step * GRAD_ACCUM * MAX_SEQ) / max(time.time() - start_time, 1)
            print(f"  [Step {step:6d}] Loss: {accum_loss:.4f} | PPL: {ppl:5.2f} | Speed: {tps:,.0f} tok/s | Elapsed: {elapsed:.1f}m", flush=True)

        if step % 500 == 0:
            save_payload = {
                "step": step,
                "model_state_dict": nexus.state_dict(),
                "config": cfg,
                "loss": accum_loss,
                "timestamp": time.time()
            }
            torch.save(save_payload, out_latest)
            print(f"  [Checkpoint] Periodic save written to {out_latest.name}", flush=True)

            if accum_loss < best_loss:
                best_loss = accum_loss
                torch.save(save_payload, out_best)
                print(f"  [*] New Best Checkpoint saved to {out_best.name} (Loss: {best_loss:.4f})", flush=True)

            run_probes(nexus, tokenizer, device, step)

if __name__ == "__main__":
    main()
