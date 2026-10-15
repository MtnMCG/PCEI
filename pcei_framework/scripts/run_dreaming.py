#!/usr/bin/env python3
"""
scripts/run_dreaming.py

Dedicated Phase 3 Dreaming Chamber script for PCEI.
Executes internal self-resonance and consolidation on the AmbientField slow-state
memory without external data, calcifying neural plasticity before fine-tuning.
"""

import sys
import math
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
from src.modules.plasticity import PlasticLayer
from src.engine.diffusion_utils import diffusion_training_step


def get_mean_confidence(nexus) -> float:
    conf_vals = []
    for _, module in nexus.core.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)


def main():
    parser = argparse.ArgumentParser(description="PCEI Dreaming Chamber Runner")
    parser.add_argument("--checkpoint-name", type=str, default="pcei_750m_pretrain_final.pt")
    parser.add_argument("--output-name", type=str, default="pcei_750m_dreamed_base.pt")
    parser.add_argument("--steps", type=int, default=3000, help="Number of dreaming steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between eval checks")
    parser.add_argument("--lr", type=float, default=1.0e-5, help="Learning rate for dreaming")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print("=" * 70)
    print("  PCEI PHASE 3 DREAMING CHAMBER")
    print(f"  Target Device: {device.upper()} | Base: {args.checkpoint_name}")
    print(f"  Steps: {args.steps} | LR: {args.lr:.2e}")
    print("=" * 70)

    config = PCEIConfig.load()
    ckpt_path = PROJECT_ROOT / "checkpoints" / args.checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    nexus = PCEINexus(config=config, device=device)
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=config.weight_decay)

    print(f"[Checkpoint] Loading weights from {ckpt_path.name}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
    nexus.load_ambient_state(checkpoint)
    nexus.to(nexus.system_dtype)

    if 'active_bonds' in checkpoint and hasattr(nexus.core, 'broker'):
        nexus.core.broker.active_bonds = checkpoint['active_bonds']
        print(f"[Broker] Restored {len(nexus.core.broker.active_bonds)} active bonds.")

    nexus.set_mode("ACTIVE")
    nexus.train()

    # Load held-out evaluation text to verify stability
    eval_path = PROJECT_ROOT / "data" / "wikitext_valid.txt"
    eval_tokens = []
    tokenizer = PCEITokenizer()
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            raw_eval = f.read(50000)
            eval_tokens = tokenizer.encode(raw_eval, pad=False)[:2048].unsqueeze(0).to(device)
            print(f"[Eval] Loaded {eval_tokens.size(1)} held-out evaluation tokens from wikitext_valid.txt")

    def run_eval():
        if len(eval_tokens) == 0:
            return 0.0, 0.0
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
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels).sum().item()
            acc = correct / max(shift_labels.numel(), 1)
        nexus.train()
        return loss.item(), acc

    print("\n[Dreaming] Measuring initial held-out baseline...")
    base_loss, base_acc = run_eval()
    print(f"  Baseline Held-Out Loss: {base_loss:.4f} | Accuracy: {base_acc*100:.2f}%")
    print(f"  Baseline Slow-Field Confidence: {get_mean_confidence(nexus):.4f}\n")

    print("[Dreaming] Entering Chamber...")
    t0 = time.time()
    accum_loss = 0.0

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        if nexus.ambient_field.slow_state is not None:
            target_state = nexus.ambient_field.slow_state.detach().to(device)
            dream_target = target_state.view(1, 1, -1).expand(1, 64, -1).clone()

            # Temporal variance injection
            temporal_noise = torch.randn_like(dream_target) * (dream_target.std() * 0.2 + 1e-4)
            dream_target = dream_target + temporal_noise

            total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                nexus, dream_target, max_T=12, use_min_snr=False
            )
            total_loss.backward()

            nexus.neurogenesis.monitor_gradients()
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()

            # Metabolic loop consolidation
            nexus.ambient_field.tick(dream_target.detach(), mse_val)
            nexus.adjust_plasticity(optimizer)

            accum_loss += mse_val

        # Physics tick
        nexus.core.broker.tick_decay()
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
        nexus.ambient_field.tick_consolidation()

        if (step + 1) % args.eval_every == 0 or step == 0:
            avg_mse = accum_loss / (args.eval_every if step > 0 else 1)
            accum_loss = 0.0
            ev_loss, ev_acc = run_eval()
            bonds = len(nexus.core.broker.active_bonds)
            conf = get_mean_confidence(nexus)
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            print(
                f"[Dream {step+1:04d}/{args.steps}] Denoise Loss: {avg_mse:.6f} | "
                f"Held-Out Acc: {ev_acc*100:.2f}% (Loss: {ev_loss:.4f}) | "
                f"Conf: {conf:.4f} | Bonds: {bonds} | {steps_per_sec:.1f} steps/s",
                flush=True
            )

    out_path = PROJECT_ROOT / "checkpoints" / args.output_name
    print(f"\n[Checkpoint] Saving post-dreaming base checkpoint to {out_path.name}...")
    state = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ambient_slow_state': nexus.ambient_field.slow_state,
        'ambient_fast_state': nexus.ambient_field.fast_state,
        'ambient_layer_emas': nexus.ambient_field.layer_emas,
        'active_bonds': dict(nexus.core.broker.active_bonds),
        'global_step': checkpoint.get('global_step', 168501) + args.steps,
        'config': config,
    }
    PCEINexus.save_safe_checkpoint(nexus, state, out_path)

    total_time = time.time() - t0
    print("=" * 70)
    print(f"  DREAMING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)!")
    print(f"  Final Held-Out Accuracy: {ev_acc*100:.2f}% | Slow Field Confidence: {conf:.4f}")
    print(f"  Saved Base: checkpoints/{args.output_name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
