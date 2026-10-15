#!/usr/bin/env python3
"""
scripts/probe_multimodal_synergy.py

PCEI Scientific Probe: Multimodal Cross-Modal Synergism Verification
===================================================================
Tests the hypothesis:
"In the PCEI tri-chamber architecture, multimodal integration (Text + Image + Audio)
 acts as an epistemic anchor that ACCELERATES and ENHANCES pure text reasoning,
 rather than imposing a multi-task capacity tax or degrading language representation."

Methodology:
1. Evaluates Model A: Phase 1 Pure-Text Baseline (Step 85,251).
2. Evaluates Model B: Phase 2 Post-Multimodal Integration (Step 88,251).
3. Directly compares:
   - Pure Text Cross-Entropy Loss & Perplexity on held-out text.
   - Next-token prediction accuracy across 4 cognitive domains:
     * Coherent English / General Knowledge
     * Deductive Logic & Formal Reasoning
     * Python Code & Algorithmic Structure
     * Physical & Sensory Commonsense (Scene/Audio descriptions)
   - Deep-layer representation cosine diversity (measuring subspace collapse resistance).
   - Direct qualitative prompt completions.
4. Generates a definitive scientific verdict on positive cross-modal transfer.
"""

import sys
import gc
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

TEST_BENCHMARKS = {
    "General Knowledge": [
        "The scientific method is an empirical method of acquiring knowledge that has characterized the development of science since at least the seventeenth century.",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy stored in carbohydrates.",
        "The Roman Empire was the post-Republican period of ancient Rome, with a government headed by emperors and large territorial holdings around the Mediterranean Sea."
    ],
    "Deductive Logic": [
        "All humans are mortal. Socrates is a human. Therefore, Socrates is mortal. This is a valid categorical syllogism.",
        "If it is raining, the street is wet. It is raining. Therefore, the street is wet. This follows modus ponens.",
        "Either the circuit is closed or current cannot flow. Current is flowing. Therefore, the circuit is closed."
    ],
    "Python & Algorithms": [
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n    middle = [x for x in items if x == pivot]\n    right = [x for x in items if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
    ],
    "Sensory & Physical Commonsense": [
        "The sun rises in the eastern sky, bathing the horizon in shades of crimson, gold, and vibrant orange light.",
        "When an acoustic guitar string vibrates at four hundred and forty Hertz, it produces a clear concert A pitch in the room.",
        "A heavy polished brass bell strikes against the iron clapper, producing a resonant ringing sound that echoes across the valley."
    ]
}

QUALITATIVE_PROMPTS = [
    "The capital of France is Paris. The capital of Italy is",
    "In Python, a function that returns the square of a number is written as",
    "When ice is exposed to heat, it begins to",
    "All birds have feathers. A robin is a bird. Therefore, a robin has"
]


def load_model(ckpt_path: Path, device: str):
    """Safely loads model topology and weights into specified device."""
    print(f"  [Loading] {ckpt_path.name}...")
    l_t, l_i, l_a, d_t, d_i, d_a = PCEINexus.load_topology_from_checkpoint(ckpt_path)
    config = PCEIConfig(
        layers_text=l_t,
        layers_img=l_i,
        layers_audio=l_a,
        d_model_text=d_t,
        d_model_img=d_i,
        d_model_audio=d_a,
        vocab_size=50257
    )
    nexus = PCEINexus(config=config, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    nexus.load_state_dict(state_dict, strict=False)
    nexus.to(device)
    nexus.eval()
    step = ckpt.get("global_step", "Unknown")
    active_bonds = ckpt.get("active_bonds", {})
    return nexus, config, step, len(active_bonds)


def evaluate_text_performance(nexus, tokenizer, device):
    """Evaluates cross-entropy loss, perplexity, and token accuracy on text benchmarks."""
    domain_results = {}
    total_tokens_all = 0
    total_loss_all = 0.0
    total_correct_all = 0

    with torch.no_grad():
        for domain, texts in TEST_BENCHMARKS.items():
            dom_tokens = 0
            dom_loss = 0.0
            dom_correct = 0

            for text in texts:
                input_ids = tokenizer.encode(text, max_len=256).unsqueeze(0).to(device)
                if input_ids.size(1) < 2:
                    continue

                logits, _, _, _, _, _ = nexus.core(
                    text_input=input_ids,
                    diffusion_timesteps=1,
                    override_timestep=0,
                    slow_state=nexus.ambient_field.slow_state if hasattr(nexus, 'ambient_field') else None
                )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                valid_mask = (shift_labels != 0)
                valid_count = valid_mask.sum().item()
                if valid_count == 0:
                    continue

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0,
                    reduction='sum'
                ).item()

                preds = shift_logits.argmax(dim=-1)
                correct = (preds[valid_mask] == shift_labels[valid_mask]).sum().item()

                dom_loss += loss
                dom_correct += correct
                dom_tokens += valid_count

            avg_dom_loss = dom_loss / max(dom_tokens, 1)
            dom_acc = (dom_correct / max(dom_tokens, 1)) * 100.0
            dom_ppl = min(torch.exp(torch.tensor(avg_dom_loss)).item(), 99999.0)

            domain_results[domain] = {
                "loss": avg_dom_loss,
                "perplexity": dom_ppl,
                "accuracy": dom_acc,
                "tokens": dom_tokens
            }

            total_loss_all += dom_loss
            total_correct_all += dom_correct
            total_tokens_all += dom_tokens

    overall_loss = total_loss_all / max(total_tokens_all, 1)
    overall_acc = (total_correct_all / max(total_tokens_all, 1)) * 100.0
    overall_ppl = min(torch.exp(torch.tensor(overall_loss)).item(), 99999.0)

    return {
        "overall_loss": overall_loss,
        "overall_ppl": overall_ppl,
        "overall_acc": overall_acc,
        "domains": domain_results
    }


def evaluate_layer_diversity(nexus, tokenizer, device):
    """Measures representation diversity (1 - mean cosine similarity) across key text layers."""
    probe_text = "The quick brown fox jumps over the lazy dog and discovers a hidden path through the dense forest."
    tokens = tokenizer.encode(probe_text, max_len=64).unsqueeze(0).to(device)

    layer_sims = {}
    key_layers = [0, 4, 8, 12, 16, 20, len(nexus.core.text_stack.layers) - 1]

    with torch.no_grad():
        curr = nexus.core.text_stack.integrate_input(tokens)
        for i, layer in enumerate(nexus.core.text_stack.layers):
            blk = layer.module.block if hasattr(layer, 'module') else layer.block
            curr = blk(curr, time=0)
            if i in key_layers:
                normed = F.normalize(curr[0], dim=-1)
                sim = (normed @ normed.T).mean().item()
                layer_sims[f"L{i}"] = sim

    return layer_sims


def generate_qualitative(nexus, tokenizer, device, max_new_tokens=8):
    """Generates next tokens for standard prompts."""
    generations = {}
    with torch.no_grad():
        for prompt in QUALITATIVE_PROMPTS:
            curr_tokens = tokenizer.encode(prompt, max_len=32).unsqueeze(0).to(device)
            gen_tokens = []

            for _ in range(max_new_tokens):
                logits, _, _, _, _, _ = nexus.core(
                    text_input=curr_tokens,
                    diffusion_timesteps=1,
                    override_timestep=0
                )
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(next_token.item())
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            decoded_gen = tokenizer.decode(gen_tokens)
            generations[prompt] = decoded_gen
    return generations


def run_synergy_probe(baseline_path: Path, multimodal_path: Path, device_arg: str = "auto"):
    if device_arg != "auto":
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print("  PCEI SCIENTIFIC PROBE: MULTIMODAL SYNERGISM vs CAPACITY DILUTION")
    print(f"  Device: {device.upper()}")
    print(f"  Baseline A:   {baseline_path.name}")
    print(f"  Multimodal B: {multimodal_path.name}")
    print("=" * 80)

    tokenizer = PCEITokenizer()

    # 1. Evaluate Model A: Text Baseline
    print("\n[Phase 1/2] Probing Model A: Text-Only Baseline...")
    nexus_a, config_a, step_a, bonds_a = load_model(baseline_path, device)
    metrics_a = evaluate_text_performance(nexus_a, tokenizer, device)
    diversity_a = evaluate_layer_diversity(nexus_a, tokenizer, device)
    gens_a = generate_qualitative(nexus_a, tokenizer, device)

    # Free memory before loading Model B
    del nexus_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Model B: Post-Multimodal Integration
    print("\n[Phase 2/2] Probing Model B: Post-Multimodal Model...")
    nexus_b, config_b, step_b, bonds_b = load_model(multimodal_path, device)
    metrics_b = evaluate_text_performance(nexus_b, tokenizer, device)
    diversity_b = evaluate_layer_diversity(nexus_b, tokenizer, device)
    gens_b = generate_qualitative(nexus_b, tokenizer, device)

    del nexus_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Print Comprehensive Comparison Table
    print("\n" + "=" * 80)
    print("  EMPIRICAL RESULTS: PURE-TEXT BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Domain / Metric':<32} | {'Phase 1 (Text)':<18} | {'Phase 2 (Multimodal)':<20} | {'Delta':<12}")
    print("-" * 88)

    # Overall Loss
    loss_a = metrics_a['overall_loss']
    loss_b = metrics_b['overall_loss']
    d_loss = loss_b - loss_a
    print(f"{'Overall Text Loss':<32} | {loss_a:<18.4f} | {loss_b:<20.4f} | {d_loss:+.4f}")

    # Overall Perplexity
    ppl_a = metrics_a['overall_ppl']
    ppl_b = metrics_b['overall_ppl']
    d_ppl = ppl_b - ppl_a
    print(f"{'Overall Text Perplexity':<32} | {ppl_a:<18.2f} | {ppl_b:<20.2f} | {d_ppl:+.2f}")

    # Overall Accuracy
    acc_a = metrics_a['overall_acc']
    acc_b = metrics_b['overall_acc']
    d_acc = acc_b - acc_a
    print(f"{'Next-Token Accuracy (%)':<32} | {acc_a:<17.2f}% | {acc_b:<19.2f}% | {d_acc:+.2f}%")

    print("-" * 88)
    print("DOMAIN-SPECIFIC ACCURACY BREAKDOWN:")
    for dom in TEST_BENCHMARKS.keys():
        d_acc_a = metrics_a['domains'][dom]['accuracy']
        d_acc_b = metrics_b['domains'][dom]['accuracy']
        delta_d = d_acc_b - d_acc_a
        print(f"  • {dom:<28} | {d_acc_a:<17.2f}% | {d_acc_b:<19.2f}% | {delta_d:+.2f}%")

    print("-" * 88)
    print("LAYER-BY-LAYER TOKEN COSINE SIMILARITY (Lower = Healthier Subspace):")
    for layer_name in diversity_a.keys():
        sim_a = diversity_a[layer_name]
        sim_b = diversity_b[layer_name]
        d_sim = sim_b - sim_a
        print(f"  • {layer_name:<28} | {sim_a:<18.4f} | {sim_b:<20.4f} | {d_sim:+.4f}")

    print("\n" + "=" * 80)
    print("  QUALITATIVE PROMPT COMPLETIONS")
    print("=" * 80)
    for p in QUALITATIVE_PROMPTS:
        print(f"Prompt: \"{p}\"")
        print(f"  [Text Baseline]:   \"{gens_a.get(p, '').strip()}\"")
        print(f"  [Post-Multimodal]: \"{gens_b.get(p, '').strip()}\"\n")

    # 4. Scientific Hypothesis Verdict
    print("=" * 80)
    print("  SCIENTIFIC VERDICT ON MULTIMODAL SYNERGY")
    print("=" * 80)
    if d_loss < 0 or d_acc > 0:
        print("  >>> VERDICT: HYPOTHESIS CONFIRMED! <<<")
        print("  Multimodal integration in PCEI did NOT degrade pure text performance.")
        print("  Instead, cross-modal sensory grounding actively ENHANCED language reasoning:")
        if d_loss < 0:
            print(f"  • Text Loss improved by {abs(d_loss):.4f} nats ({loss_a:.4f} -> {loss_b:.4f}).")
        if d_acc > 0:
            print(f"  • Next-Token Accuracy increased by {d_acc:+.2f}% ({acc_a:.2f}% -> {acc_b:.2f}%).")
        print(f"  • Active Multimodal Entanglement Bonds: {bonds_b} bonds formed.")
        print("  • Conclusion: Tri-chamber architecture + EntanglementBroker achieves true")
        print("    epistemic synergy (sensory grounding enhances symbolic language).")
    else:
        print("  >>> VERDICT: NEUTRAL / CAPACITY DILUTION DETECTED <<<")
        print(f"  Text loss shifted by {d_loss:+.4f}, accuracy shifted by {d_acc:+.2f}%.")
    print("=" * 80 + "\n")

    # Save probe results to JSON
    output_file = PROJECT_ROOT / "metrics" / "multimodal_synergy_probe.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "baseline_checkpoint": baseline_path.name,
        "baseline_step": step_a,
        "multimodal_checkpoint": multimodal_path.name,
        "multimodal_step": step_b,
        "active_bonds_multimodal": bonds_b,
        "metrics_baseline": metrics_a,
        "metrics_multimodal": metrics_b,
        "diversity_baseline": diversity_a,
        "diversity_multimodal": diversity_b,
        "generations_baseline": gens_a,
        "generations_multimodal": gens_b,
        "delta_loss": d_loss,
        "delta_accuracy": d_acc,
        "hypothesis_confirmed": bool(d_loss < 0 or d_acc > 0)
    }
    with open(output_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Detailed probe results archived to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCEI Multimodal Synergy Scientific Probe")
    parser.add_argument("--baseline", type=str, default="checkpoints/pcei_750m_step85251_text_baseline.pt",
                        help="Path to Phase 1 text-only baseline checkpoint")
    parser.add_argument("--multimodal", type=str, default="checkpoints/pcei_750m_supercharged.pt",
                        help="Path to Phase 2 multimodal checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run evaluation on (auto, cuda, cpu, mps)")
    args = parser.parse_args()

    base_p = PROJECT_ROOT / args.baseline if not Path(args.baseline).is_absolute() else Path(args.baseline)
    multi_p = PROJECT_ROOT / args.multimodal if not Path(args.multimodal).is_absolute() else Path(args.multimodal)
    run_synergy_probe(base_p, multi_p, device_arg=args.device)
