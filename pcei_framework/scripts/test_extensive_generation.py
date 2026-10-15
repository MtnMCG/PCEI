#!/usr/bin/env python3
"""
scripts/test_extensive_generation.py

Runs extensive, multi-paragraph generation tests on PCEI checkpoints (e.g. pcei_750m_supercharged.pt).
Tests reasoning, science, creative writing, programming, and philosophical dialogue
with full token outputs (120-250 tokens per prompt) and windowed repetition penalty.
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

DEFAULT_CHECKPOINT = "checkpoints/pcei_750m_supercharged.pt"

PROMPTS = [
    {
        "domain": "Identity & Architecture",
        "prompt": "USER: Hello! What is your name and what model architecture are you?\nASSISTANT:",
        "max_new": 140,
    },
    {
        "domain": "Elementary Arithmetic",
        "prompt": "USER: What is 2 + 2, and what is 15 - 8?\nASSISTANT:",
        "max_new": 90,
    },
    {
        "domain": "Creative Storytelling",
        "prompt": "USER: Write a descriptive scene about an astronomer working late at night in a mountaintop observatory when the telescope detects an unexpected signal.\nASSISTANT:",
        "max_new": 180,
    },
    {
        "domain": "Scientific Explanation",
        "prompt": "USER: Why does ice float on liquid water, and why is this property essential for aquatic life in winter?\nASSISTANT:",
        "max_new": 180,
    },
    {
        "domain": "Deductive Logic & Reasoning",
        "prompt": "USER: Read these premises:\n1. All mammals are warm-blooded.\n2. All dolphins are mammals.\n3. Flipper is a dolphin.\nWhat can we conclude about Flipper? Explain your reasoning step by step.\nASSISTANT:",
        "max_new": 160,
    },
    {
        "domain": "Code Generation & Python",
        "prompt": "USER: Write a clean Python function to find the maximum value in a list of numbers without using the built-in max() function.\nASSISTANT:",
        "max_new": 160,
    },
    {
        "domain": "Philosophy & Epistemology",
        "prompt": "USER: What did René Descartes mean by 'Cogito, ergo sum', and what problem was he trying to solve?\nASSISTANT:",
        "max_new": 180,
    },
    {
        "domain": "Atmospheric Prose Continuation",
        "prompt": "The ancient library was buried beneath three centuries of dust and desert sand, yet when the explorers broke the brass seal on the heavy iron door,",
        "max_new": 160,
    },
]


def generate_streaming(
    nexus,
    tokenizer,
    prompt: str,
    max_new: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
    rep_penalty: float = 1.15,
    rep_window: int = 50,
    device: str = "cpu",
    stream: bool = True
) -> str:
    """
    Generates text autoregressively with windowed repetition penalty and streaming output.
    """
    ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    curr = ids.clone()
    generated_tokens = []
    
    stop_phrases = ["\nUSER:", "\nHuman:", "\nQuestion:", "<|endoftext|>"]
    accumulated_text = ""

    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new):
            logits, *_ = nexus.core(
                text_input=curr,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
            )
            logits_last = logits[:, -1, :].squeeze(0).clone()

            # Windowed repetition penalty: penalize only recent tokens to prevent loops
            # while allowing natural reuse of words across paragraphs.
            if rep_penalty != 1.0 and len(generated_tokens) > 0:
                penalty_tokens = set(curr[0, -rep_window:].tolist())
                for t in penalty_tokens:
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
            new_piece = tokenizer.decode([next_tok])
            accumulated_text += new_piece

            if stream:
                sys.stdout.write(new_piece)
                sys.stdout.flush()

            # Check for conversational stop phrases
            if any(accumulated_text.endswith(sp) for sp in stop_phrases):
                break

            curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)

    t1 = time.perf_counter()
    n_tok = len(generated_tokens)
    speed = n_tok / (t1 - t0) if (t1 - t0) > 0 else 0
    if stream:
        sys.stdout.write(f"\n   [{n_tok} tokens in {t1 - t0:.2f}s | {speed:.1f} tok/s]\n")
        sys.stdout.flush()

    return accumulated_text


def run_extensive_tests(ckpt_path: Path, device: str, temp: float, top_p: float, rep_pen: float, prompt_idx: int = None):
    print("=" * 80)
    print("  PCEI EXTENSIVE GENERATION TEST SUITE")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Target Device: {device.upper()} | Temp: {temp} | Top-P: {top_p} | Rep Penalty: {rep_pen}")
    print("=" * 80)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print("\n[1/3] Loading topology from checkpoint...")
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
    config = PCEIConfig(
        layers_text=l_t, layers_img=l_i, layers_audio=l_a,
        d_model_text=d_t, d_model_img=d_i, d_model_audio=d_a,
        vocab_size=50257
    )
    print(f"      Topology: {l_t} Text Layers (D={d_t}) | {l_i} Img Layers | {l_a} Audio Layers")

    print("[2/3] Initializing PCEINexus & loading weights...")
    nexus = PCEINexus(config=config, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    nexus.to(device)
    nexus.eval()

    total_p = sum(p.numel() for p in nexus.parameters())
    step = ckpt.get("global_step", "Unknown")
    print(f"      Parameters: {total_p:,} | Global Training Step: {step}")

    tokenizer = PCEITokenizer()

    test_items = PROMPTS if prompt_idx is None else [PROMPTS[prompt_idx]]

    print("\n[3/3] Generating extensive responses across domains...\n")
    results = []
    for i, item in enumerate(test_items, 1):
        domain = item["domain"]
        prompt = item["prompt"]
        max_new = item["max_new"]

        print("─" * 80)
        print(f"  PROMPT {i}/{len(test_items)}: [{domain}]")
        print(f"  INPUT:\n{prompt}")
        print("  GENERATED RESPONSE:")

        resp = generate_streaming(
            nexus=nexus,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new=max_new,
            temperature=temp,
            top_p=top_p,
            rep_penalty=rep_pen,
            rep_window=50,
            device=device,
            stream=True
        )
        results.append((domain, prompt, resp))
        print()

    print("=" * 80)
    print("  ALL EXTENSIVE GENERATION TESTS COMPLETED")
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="PCEI Extensive Generation Evaluation")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--rep-pen", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--prompt-idx", type=int, default=None, help="Optional single prompt index to run")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = PROJECT_ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    run_extensive_tests(
        ckpt_path=ckpt_path,
        device=device,
        temp=args.temp,
        top_p=args.top_p,
        rep_pen=args.rep_pen,
        prompt_idx=args.prompt_idx
    )


if __name__ == "__main__":
    main()
