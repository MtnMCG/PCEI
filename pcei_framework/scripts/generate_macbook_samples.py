#!/usr/bin/env python3
"""
scripts/generate_macbook_samples.py

Interactive & Automated Text Generation Probe for PCEI on Apple Silicon (MPS/CPU)
================================================================================
Loads the trained checkpoint, runs autoregressive generation across diverse prompts
with greedy decoding and top-p sampling (with repetition penalty), and reports
detailed token probability distributions and generation text.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

TEST_PROMPTS = [
    ("Greeting", "Hi there! Who are you and how are you feeling today?"),
    ("Geography", "The capital of France is Paris. The capital of Italy is"),
    ("Physical Commonsense", "When water is heated to 100 degrees Celsius, it begins to"),
    ("Deductive Logic", "All humans are mortal. Socrates is a human. Therefore, Socrates is"),
    ("Python Coding", "def add_numbers(a, b):\n    # Return the sum of a and b\n    return")
]


def generate(nexus, tokenizer, prompt: str, max_new_tokens: int = 24, temperature: float = 0.7, top_p: float = 0.85, repetition_penalty: float = 1.2, device: str = "mps"):
    input_ids = tokenizer.encode(prompt, max_len=64).unsqueeze(0).to(device)
    curr_tokens = input_ids.clone()
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _, _, _, _ = nexus.core(
                text_input=curr_tokens,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state if hasattr(nexus, 'ambient_field') else None
            )
            next_token_logits = logits[:, -1, :].clone().squeeze(0)  # (V,)

            # Repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for prev_tok in set(curr_tokens[0].tolist()):
                    if next_token_logits[prev_tok] > 0:
                        next_token_logits[prev_tok] /= repetition_penalty
                    else:
                        next_token_logits[prev_tok] *= repetition_penalty

            if temperature == 0.0:
                next_tok = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
            else:
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float("Inf")
                    probs = F.softmax(scaled_logits, dim=-1)

                next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            tok_id = next_tok.item()
            generated.append(tok_id)
            curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)

            if tok_id == tokenizer.pad_id or tok_id == 50256:  # EOS / EOT
                break

    return tokenizer.decode(generated)


def get_topk_next_tokens(nexus, tokenizer, prompt: str, k: int = 5, device: str = "mps"):
    input_ids = tokenizer.encode(prompt, max_len=64).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _, _, _, _ = nexus.core(
            text_input=input_ids,
            diffusion_timesteps=1,
            override_timestep=0,
            slow_state=nexus.ambient_field.slow_state if hasattr(nexus, 'ambient_field') else None
        )
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, k)
        results = []
        for idx, prob in zip(topk.indices, topk.values):
            decoded_tok = tokenizer.decode([idx.item()])
            results.append((decoded_tok, prob.item()))
    return results


def main():
    parser = argparse.ArgumentParser(description="PCEI MacBook Local Generation Test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcei_750m_3phase_completed.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device (mps, cpu, auto)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    ckpt_path = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    print("=" * 75)
    print(f"  PCEI 750M MACBOOK GENERATION PROBE")
    print(f"  Device: {device.upper()} | Checkpoint: {ckpt_path.name}")
    print("=" * 75)

    # 1. Load Model
    print("[1/3] Loading Model Topology...")
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
    config = PCEIConfig(
        layers_text=l_t, layers_img=l_i, layers_audio=l_a,
        d_model_text=d_t, d_model_img=d_i, d_model_audio=d_a,
        vocab_size=50257
    )
    nexus = PCEINexus(config=config, device=device)

    print("[2/3] Loading Weights & State Dict...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    nexus.load_state_dict(state_dict, strict=False)
    nexus.to(device)
    nexus.eval()

    step = ckpt.get("global_step", "Unknown")
    total_params = sum(p.numel() for p in nexus.parameters())
    print(f"      Model Ready: {total_params:,} parameters | Global Step: {step}")

    # 2. Run Generation Suite
    print("\n[3/3] Running Generation Suite Across Diverse Prompts...")
    tokenizer = PCEITokenizer()

    for category, prompt in TEST_PROMPTS:
        print("\n" + "─" * 70)
        print(f"Category: {category}")
        print(f"Prompt:   \"{prompt}\"")

        # Top 5 next token predictions
        topk = get_topk_next_tokens(nexus, tokenizer, prompt, k=5, device=device)
        topk_str = ", ".join([f"{repr(t)}: {p*100:.1f}%" for t, p in topk])
        print(f"Top-5 Next Tokens: {topk_str}")

        # Greedy generation
        greedy_gen = generate(nexus, tokenizer, prompt, max_new_tokens=16, temperature=0.0, device=device)
        print(f"Greedy Output:     \"{greedy_gen.strip()}\"")

        # Sampled generation (temperature=0.7, top_p=0.85, repetition_penalty=1.2)
        sample_gen = generate(nexus, tokenizer, prompt, max_new_tokens=16, temperature=0.7, top_p=0.85, repetition_penalty=1.2, device=device)
        print(f"Sampled Output:    \"{sample_gen.strip()}\"")

    print("\n" + "=" * 75)
    print("  GENERATION SUITE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
