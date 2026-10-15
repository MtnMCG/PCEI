#!/usr/bin/env python3
"""
scripts/run_domain_specific_generation_tests.py
Executes domain-specific generation benchmarks across robotics/embodied AI, low-bit quantization,
neuroscience/plasticity, selective SSMs, and physics reasoning.
"""

import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

DOMAIN_PROMPTS = [
    {
        "domain": "1. Embodied Robotics & Locomotion",
        "prompt": "USER: Explain how an embodied bipedal humanoid agent navigates angled Quad Step obstacles without falling, and why non-linear spline manifolds excel in continuous torque control.\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "2. BitNet 1.58b Quantization & Low-Bit Theory",
        "prompt": "USER: How do strictly ternary weights {-1, 0, +1} in BitNet 1.58b match floating-point language modeling performance while eliminating matrix multiplication energy?\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "3. Biological Plasticity & Metabolic Pruning",
        "prompt": "USER: Describe how metabolic decay prunes weak synaptic pathways into quiescent zero-states and how offline dreaming consolidation prevents catastrophic forgetting.\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "4. Selective State Spaces (SSM) vs Transformer Attention",
        "prompt": "USER: Why do selective state spaces like Mamba process sequence context with O(L) compute and O(1) recurrent memory, avoiding Transformer KV-cache cliffs?\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "5. Continuous Physics & Nonlinear Dynamics",
        "prompt": "USER: In a physical inverted pendulum system governed by inertia, Coriolis force, and gravity, what control law stabilizes the agent at the upright vertical equilibrium?\nASSISTANT:",
        "max_new": 140,
    }
]

def generate_tokens(nexus, tokenizer, prompt, max_new=140, temperature=0.7, top_p=0.9, rep_penalty=1.15, device="cpu"):
    ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    curr = ids.clone()
    generated_tokens = []
    accumulated_text = ""
    stop_phrases = ["\nUSER:", "\nHuman:", "\nQuestion:", "<|endoftext|>"]

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new):
            logits, *_ = nexus.core(
                text_input=curr,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=getattr(nexus, "ambient_field", None) and nexus.ambient_field.slow_state,
            )
            logits_last = logits[:, -1, :].squeeze(0).clone()

            # Windowed repetition penalty
            if rep_penalty != 1.0 and len(generated_tokens) > 0:
                recent = set(curr[0, -50:].tolist())
                for t in recent:
                    if logits_last[t] > 0:
                        logits_last[t] /= rep_penalty
                    else:
                        logits_last[t] *= rep_penalty

            if temperature <= 0.01:
                next_tok = logits_last.argmax().item()
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
                next_tok = torch.multinomial(probs, 1).item()

            if next_tok in (0, 50256):
                break

            generated_tokens.append(next_tok)
            piece = tokenizer.decode([next_tok])
            accumulated_text += piece

            if any(accumulated_text.endswith(sp) for sp in stop_phrases):
                break

            curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)

    t1 = time.perf_counter()
    speed = len(generated_tokens) / (t1 - t0) if (t1 - t0) > 0 else 0
    return accumulated_text, len(generated_tokens), speed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pcei_framework/checkpoints/pcei_100m_true_ternary.pt")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"Error: checkpoint {args.checkpoint} not found")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading checkpoint: {ckpt_path.name} on {device.upper()}...")
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
    config = PCEIConfig(
        layers_text=l_t, layers_img=l_i, layers_audio=l_a,
        d_model_text=d_t, d_model_img=d_i, d_model_audio=d_a,
        vocab_size=50257
    )
    nexus = PCEINexus(config=config, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    nexus.to(device)
    nexus.eval()

    tokenizer = PCEITokenizer()
    print("\n" + "=" * 80)
    print(f"  PCEI DOMAIN-SPECIFIC GENERATION BENCHMARKS ({ckpt_path.name})")
    print(f"  Topology: {l_t} Text Layers | D={d_t} | Step: {ckpt.get("global_step", "N/A")}")
    print("=" * 80 + "\n")

    for item in DOMAIN_PROMPTS:
        domain = item["domain"]
        prompt = item["prompt"]
        max_new = item["max_new"]

        print(f"### [DOMAIN: {domain}]")
        print(f"Prompt: {prompt.strip()}")
        out_text, n_tok, speed = generate_tokens(nexus, tokenizer, prompt, max_new=max_new, device=device)
        print(f"Output:\n{out_text.strip()}")
        print(f"({n_tok} tokens generated | {speed:.1f} tokens/sec)\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
