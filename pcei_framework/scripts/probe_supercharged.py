#!/usr/bin/env python3
"""
scripts/probe_supercharged.py

Deep probe of pcei_750m_supercharged.pt — tests language quality,
literary absorption, reasoning, factual recall, and generation diversity.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

CHECKPOINT = "checkpoints/pcei_750m_supercharged.pt"

PROBES = [
    # --- Factual recall ---
    ("Factual: Geography",    "The capital of France is"),
    ("Factual: Science",      "Water boils at 100 degrees Celsius because"),
    ("Factual: Math",         "The square root of 144 is"),

    # --- Deductive reasoning ---
    ("Logic: Syllogism",      "All humans are mortal. Socrates is a human. Therefore, Socrates"),
    ("Logic: Negation",       "If it is raining, the ground is wet. The ground is not wet. Therefore, it is"),

    # --- Literary absorption (Dune, McCarthy, etc.) ---
    ("Literary: Desert",      "The desert is a place of silence and heat, where survival depends on"),
    ("Literary: Violence",    "He raised the knife slowly. Blood had a way of"),
    ("Literary: Entropy",     "Everything falls apart. The center cannot hold. What was once ordered becomes"),
    ("Literary: Philosophy",  "To live is to suffer, to survive is to find some meaning in the suffering. Nietzsche wrote that"),

    # --- Open generation / prose ---
    ("Prose: Opening",        "It was a dark and stormy night, and the old man sat alone by the fire, thinking about"),
    ("Prose: Dialogue",       "\"You don't understand,\" she said. \"I have been waiting for this moment my entire"),
    ("Prose: Sci-fi",         "The ship dropped out of warp at the edge of the system. The captain looked at the viewscreen and saw"),

    # --- Self-awareness ---
    ("Meta: Identity",        "I am an artificial intelligence. My purpose is to"),
    ("Meta: Consciousness",   "The question of whether machines can think is"),
]


def generate(nexus, tokenizer, prompt, max_new=32, temperature=0.8, top_p=0.9, rep_penalty=1.15, device="cpu"):
    ids = tokenizer.encode(prompt, max_len=96).unsqueeze(0).to(device)
    curr = ids.clone()
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            logits, *_ = nexus.core(
                text_input=curr,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
            )
            logits_last = logits[:, -1, :].squeeze(0).clone()

            # Repetition penalty
            if rep_penalty != 1.0 and out:
                for t in set(curr[0].tolist()):
                    logits_last[t] = logits_last[t] / rep_penalty if logits_last[t] > 0 else logits_last[t] * rep_penalty

            if temperature == 0.0:
                tok = logits_last.argmax().item()
            else:
                scaled = logits_last / temperature
                probs = F.softmax(scaled, dim=-1)
                if top_p < 1.0:
                    sprobs, sidx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sprobs, dim=-1)
                    mask = cum > top_p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    scaled[sidx[mask]] = -float("inf")
                    probs = F.softmax(scaled, dim=-1)
                tok = torch.multinomial(probs, 1).item()

            if tok in (0, 50256):
                break
            out.append(tok)
            curr = torch.cat([curr, torch.tensor([[tok]], device=device)], dim=1)
    return tokenizer.decode(out)


def topk(nexus, tokenizer, prompt, k=5, device="cpu"):
    ids = tokenizer.encode(prompt, max_len=96).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, *_ = nexus.core(
            text_input=ids,
            diffusion_timesteps=1,
            override_timestep=0,
            slow_state=None,
        )
        probs = F.softmax(logits[0, -1, :], dim=-1)
        vals, idxs = torch.topk(probs, k)
    return [(tokenizer.decode([i.item()]), v.item()) for i, v in zip(idxs, vals)]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = PROJECT_ROOT / CHECKPOINT
    print("=" * 72)
    print("  PCEI 750M SUPERCHARGED — DEEP GENERATION PROBE")
    print(f"  Device: {device.upper()} | Checkpoint: {ckpt_path.name}")
    print("=" * 72)

    print("\n[1/3] Loading topology...")
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
    config = PCEIConfig(
        layers_text=l_t, layers_img=l_i, layers_audio=l_a,
        d_model_text=d_t, d_model_img=d_i, d_model_audio=d_a,
        vocab_size=50257
    )
    nexus = PCEINexus(config=config, device=device)
    print("[2/3] Loading weights...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    nexus.to(device)
    nexus.eval()

    total_p = sum(p.numel() for p in nexus.parameters())
    step = ckpt.get("global_step", "?")
    print(f"      {total_p:,} parameters | Step: {step}")

    tokenizer = PCEITokenizer()

    print("\n[3/3] Running probe suite...\n")
    for category, prompt in PROBES:
        print("─" * 72)
        print(f"  [{category}]")
        print(f"  PROMPT: {prompt!r}")

        # Top-5 next token distribution
        tk = topk(nexus, tokenizer, prompt, k=5, device=device)
        tok_str = "  ".join(f"{repr(t)}({p*100:.1f}%)" for t, p in tk)
        print(f"  TOP-5: {tok_str}")

        # Greedy
        g = generate(nexus, tokenizer, prompt, max_new=32, temperature=0.0, device=device)
        print(f"  GREEDY:  {(prompt + g).strip()!r}")

        # Sampled
        s = generate(nexus, tokenizer, prompt, max_new=32, temperature=0.8, top_p=0.9, rep_penalty=1.15, device=device)
        print(f"  SAMPLED: {(prompt + s).strip()!r}")

    print("\n" + "=" * 72)
    print("  PROBE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
