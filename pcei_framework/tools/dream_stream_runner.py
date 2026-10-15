#!/usr/bin/env python3
"""
tools/dream_stream_runner.py

PCEI Dreaming Chamber with Live Distilled Thought Streaming.
Executes internal self-resonance and ambient field consolidation while
decoding recurrent latent activations into readable English dream thoughts.
Logs real-time denoise loss, perplexity, and decoded thoughts to logs/dream_stream.log.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step


def get_mean_confidence(nexus: PCEINexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)


DREAM_PROMPT_SEEDS = [
    "The physical structure of reality and consciousness begins with",
    "In deep contemplation, the relationship between logic and matter shows that",
    "When mathematical patterns emerge from continuous space,",
    "The law of thermodynamics and temperature dictates that",
    "Reasoning step by step across memory pathways reveals that",
    "Energy flows through connected topological nodes because",
    "Language and semantic understanding reflect the continuous manifold where",
]


def sample_dream_thought(
    nexus: PCEINexus,
    tokenizer: PCEITokenizer,
    device: str,
    max_tokens: int = 48,
    temperature: float = 0.85,
    seed_text: Optional[str] = None
) -> str:
    """
    Decodes an internal dream sequence from the model's text stack using
    ambient field guidance and nucleus sampling.
    """
    nexus.eval()
    if seed_text:
        seed_ids = tokenizer.encode(seed_text)
        if isinstance(seed_ids, list):
            input_ids = torch.tensor([seed_ids], device=device, dtype=torch.long)
        else:
            input_ids = seed_ids.unsqueeze(0).to(device)
    else:
        input_ids = torch.tensor([[50256]], device=device, dtype=torch.long)

    mean_conf = get_mean_confidence(nexus)
    accumulated_tokens = []

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > 256:
                input_ids = input_ids[:, -256:]

            logits, t_lat, *_ = nexus.core(
                text_input=input_ids,
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=mean_conf
            )
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)

            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.90
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            tok_id = next_token.item()

            if tok_id in (0, 50256) and len(accumulated_tokens) > 10:
                break

            accumulated_tokens.append(tok_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    nexus.train()
    decoded = tokenizer.decode(accumulated_tokens).strip().replace("\n", " ")
    return decoded


def main():
    parser = argparse.ArgumentParser(description="PCEI Dreaming Chamber with Live Thought Stream")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_750m_reasoning_assistant_latest.pt")
    parser.add_argument("--output-name", type=str, default="pcei_750m_dream_consolidated_155k.pt")
    parser.add_argument("--cycles", type=int, default=1000, help="Number of dream consolidation cycles")
    parser.add_argument("--eval-every", type=int, default=50, help="Cycles between held-out perplexity evaluations")
    parser.add_argument("--save-every", type=int, default=250, help="Cycles between checkpoint saves")
    parser.add_argument("--lr", type=float, default=1.0e-5, help="Learning rate for dream consolidation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log_path = Path(args.log_file) if args.log_file else (PROJECT_ROOT / "logs" / "dream_stream.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_stream(msg: str, to_console: bool = True):
        t_str = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{t_str}] {msg}"
        if to_console:
            print(formatted, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
            f.flush()

    log_stream("=" * 80)
    log_stream("  PCEI DREAMING CHAMBER — INTERNAL SELF-RESONANCE & METABOLIC CONSOLIDATION")
    log_stream(f"  Target Device: {device.upper()} | Source: {args.checkpoint_name}")
    log_stream(f"  Target Cycles: {args.cycles} | Base LR: {args.lr:.2e} | Stream Log: {log_path.name}")
    log_stream("=" * 80)

    config = PCEIConfig.load()
    ckpt_path = PROJECT_ROOT / "checkpoints" / args.checkpoint_name
    if not ckpt_path.exists():
        fallback = PROJECT_ROOT / "checkpoints" / "pcei_750m_reasoning_step_150000.pt"
        if fallback.exists():
            log_stream(f"[Notice] {ckpt_path.name} not found, falling back to {fallback.name}...")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio

    log_stream(f"[Topology] {layers_text} Text Layers (D={d_text}) | {layers_img} Img | {layers_audio} Audio")
    nexus = PCEINexus(config=config, device=device)
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=config.weight_decay)

    log_stream(f"[Checkpoint] Loading weights from {ckpt_path.name}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
    nexus.load_ambient_state(checkpoint)
    nexus.to(nexus.system_dtype)

    if 'active_bonds' in checkpoint and hasattr(nexus.core, 'broker'):
        nexus.core.broker.active_bonds = checkpoint['active_bonds']
        log_stream(f"[Broker] Restored {len(nexus.core.broker.active_bonds)} active bonds.")

    nexus.set_mode("ACTIVE")
    nexus.train()
    tokenizer = PCEITokenizer()

    eval_path = PROJECT_ROOT / "data" / "synthetic" / "pcei_foundation_teacher.jsonl"
    if not eval_path.exists():
        eval_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_foundation_teacher.jsonl"

    eval_tokens = None
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            raw_eval = f.read(40000)
            token_ids = tokenizer.encode(raw_eval, pad=False)
            if len(token_ids) > 1024:
                eval_tokens = token_ids[:1024].unsqueeze(0).to(device)
                log_stream(f"[Eval] Loaded {eval_tokens.size(1)} validation tokens from {eval_path.name}")

    def evaluate_held_out() -> Tuple[float, float]:
        if eval_tokens is None:
            return 3.0, 20.08
        nexus.eval()
        with torch.no_grad():
            fwd_kwargs = {
                'diffusion_timesteps': 1,
                'override_timestep': 0,
                'slow_state': nexus.ambient_field.slow_state,
                'mean_confidence': get_mean_confidence(nexus),
                'text_input': eval_tokens[:, :512]
            }
            logits, t_lat, *_ = nexus.core(**fwd_kwargs)
            out_logits = nexus.core.text_stack.decode_output(t_lat)
            shift_logits = out_logits[:, :-1, :].contiguous()
            shift_labels = eval_tokens[:, 1:512].contiguous()
            loss_val = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0).item()
            ppl_val = math.exp(min(loss_val, 20.0))
        nexus.train()
        return loss_val, ppl_val

    init_loss, init_ppl = evaluate_held_out()
    log_stream(f"[Baseline] Initial Held-Out Loss: {init_loss:.4f} | Perplexity (PPL): {init_ppl:.2f} | Conf: {get_mean_confidence(nexus):.4f}")
    log_stream("-" * 80)

    accum_denoise_loss = 0.0
    t_start = time.time()
    curr_loss = init_loss
    curr_ppl = init_ppl

    for cycle in range(1, args.cycles + 1):
        t_c0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        denoise_step_loss = 0.0
        if nexus.ambient_field.slow_state is not None:
            target_state = nexus.ambient_field.slow_state.detach().to(device)
            dream_target = target_state.view(1, 1, -1).expand(1, 64, -1).clone()

            temporal_noise = torch.randn_like(dream_target) * (dream_target.std() * 0.15 + 1e-4)
            dream_target = dream_target + temporal_noise

            total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                nexus, dream_target, max_T=12, use_min_snr=False
            )
            total_loss.backward()

            nexus.neurogenesis.monitor_gradients()
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()

            nexus.ambient_field.tick(dream_target.detach(), mse_val)
            nexus.adjust_plasticity(optimizer)
            denoise_step_loss = float(mse_val)
            accum_denoise_loss += denoise_step_loss

        nexus.core.broker.tick_decay()
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
        nexus.ambient_field.tick_consolidation()

        if cycle % args.eval_every == 0:
            curr_loss, curr_ppl = evaluate_held_out()

        seed_prompt = DREAM_PROMPT_SEEDS[(cycle - 1) % len(DREAM_PROMPT_SEEDS)]
        thought_text = sample_dream_thought(
            nexus, tokenizer, device, max_tokens=52, temperature=0.85, seed_text=seed_prompt
        )

        bonds_count = len(nexus.core.broker.active_bonds)
        conf_val = get_mean_confidence(nexus)
        dt_cycle = max(time.time() - t_c0, 1e-3)

        thought_header = (
            f"[Cycle {cycle:04d}/{args.cycles}] Denoise Loss: {denoise_step_loss:.6f} | "
            f"Held-Out PPL: {curr_ppl:.2f} (Loss: {curr_loss:.4f}) | "
            f"Conf: {conf_val:.4f} | Bonds: {bonds_count} | Cycle: {dt_cycle:.2f}s"
        )
        thought_body = f"DREAM: \"{thought_text}\""

        log_stream(thought_header, to_console=True)
        log_stream(thought_body, to_console=True)

        if cycle % args.save_every == 0:
            inter_name = f"pcei_750m_dream_cycle_{cycle}.pt"
            inter_path = PROJECT_ROOT / "checkpoints" / inter_name
            state = {
                'model_state_dict': nexus.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ambient_slow_state': nexus.ambient_field.slow_state,
                'ambient_fast_state': nexus.ambient_field.fast_state,
                'ambient_layer_emas': nexus.ambient_field.layer_emas,
                'active_bonds': dict(nexus.core.broker.active_bonds),
                'global_step': checkpoint.get('global_step', 155000) + cycle,
                'config': config,
            }
            PCEINexus.save_safe_checkpoint(nexus, state, inter_path)
            log_stream(f"[Checkpoint] Saved intermediate dreaming checkpoint: {inter_name}")

    final_out_path = PROJECT_ROOT / "checkpoints" / args.output_name
    latest_path = PROJECT_ROOT / "checkpoints" / "pcei_750m_reasoning_assistant_latest.pt"
    log_stream(f"\n[Final] Saving consolidated dreaming state to {final_out_path.name} & {latest_path.name}...")

    final_state = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ambient_slow_state': nexus.ambient_field.slow_state,
        'ambient_fast_state': nexus.ambient_field.fast_state,
        'ambient_layer_emas': nexus.ambient_field.layer_emas,
        'active_bonds': dict(nexus.core.broker.active_bonds),
        'global_step': checkpoint.get('global_step', 155000) + args.cycles,
        'config': config,
    }
    PCEINexus.save_safe_checkpoint(nexus, final_state, final_out_path)
    PCEINexus.save_safe_checkpoint(nexus, final_state, latest_path)

    sync_signal_file = PROJECT_ROOT / "checkpoints" / "latest_sync_ready.txt"
    with open(sync_signal_file, "w") as f:
        f.write(f"STEP=dream_consolidated_{args.cycles}\nFILE={final_out_path.name}\nTIME={time.time()}\n")

    total_time = time.time() - t_start
    log_stream("=" * 80)
    log_stream(f"  DREAMING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)!")
    log_stream(f"  Final Held-Out Perplexity: {curr_ppl:.2f} (Loss: {curr_loss:.4f})")
    log_stream(f"  Final Slow-Field Confidence: {get_mean_confidence(nexus):.4f}")
    log_stream(f"  Consolidated Checkpoint Ready: {final_out_path.name}")
    log_stream("=" * 80)


if __name__ == "__main__":
    main()
