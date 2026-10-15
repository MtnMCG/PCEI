"""
pcei_framework/scripts/run_one_hour_ternary_training.py

1-Hour True BitNet 1.58b Retraining / Fine-Tuning Harness.
Trains PCEINexus (100M foundation) with strict {-1.0, 0.0, +1.0} ternary quantization
for exactly 1 hour (3,600 seconds), adapting continuous shadow weights to true 3-state
dynamics with continuous safe atomic checkpointing.
"""

import sys
import os
import time
import math
import signal
import json
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.modules.plasticity import PlasticLayer
from src.modules.ternary import TernaryLinear, TernaryQuantize
from scripts.train import (
    load_text_data,
    PCEITokenizer,
    ContinuousSequencePacker,
    diffusion_training_step
)

TARGET_DURATION_SECONDS = 3600  # Exactly 1 hour
MAX_SEQ_LEN = 128
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

def main():
    print("=" * 70)
    print("  PCEI TRUE BITNET 1.58b — 1-HOUR ADAPTATION TRAINING RUN")
    print(f"  Target Duration: {TARGET_DURATION_SECONDS} seconds (1.00 hour)")
    print(f"  Quantizer: Strictly 3-State {-1.0, 0.0, +1.0}")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    # 1. Load Checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "pcei_100m.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)

    print(f"[Checkpoint] Resuming from: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_obj = ckpt.get("config", None)
    if isinstance(cfg_obj, PCEIConfig):
        config = cfg_obj
    elif isinstance(cfg_obj, dict) and cfg_obj:
        config = PCEIConfig(**cfg_obj)
    else:
        config = PCEIConfig.load()

    nexus = PCEINexus(config=config).to(device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.train()
    nexus.set_mode("WITNESS")

    # Audit Initial Ternary Distribution
    total_q, neg1, zero0, pos1 = 0, 0, 0, 0
    for name, m in nexus.named_modules():
        if isinstance(m, TernaryLinear):
            q = m.quantized_weight
            total_q += q.numel()
            neg1 += (q == -1.0).sum().item()
            zero0 += (q == 0.0).sum().item()
            pos1 += (q == 1.0).sum().item()

    print(f"[Initial Weights] Audited {total_q:,} quantized synapses:")
    print(f"  • Inhibitory (-1): {neg1:,} ({neg1/total_q*100:.1f}%)")
    print(f"  • Pruned      (0): {zero0:,} ({zero0/total_q*100:.1f}%)")
    print(f"  • Excitatory (+1): {pos1:,} ({pos1/total_q*100:.1f}%)")

    # 2. Setup Data Pipeline
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    raw_text_dir = PROJECT_ROOT / "data" / "raw" / "text"
    curriculum_dir = PROJECT_ROOT / "data" / "curriculum"
    high_quality_dir = PROJECT_ROOT / "data" / "high_quality"

    all_texts = load_text_data(
        synthetic_dir=synthetic_dir,
        raw_text_dir=raw_text_dir,
        pure_synthetic=False,
        high_quality_only=False,
        curriculum_dir=curriculum_dir,
        high_quality_dir=high_quality_dir
    )
    random.shuffle(all_texts)
    print(f"[Dataset] Active corpus size: {len(all_texts):,} sequences")

    tokenizer = PCEITokenizer(vocab_size=config.vocab_size)
    packer = ContinuousSequencePacker(all_texts, tokenizer, max_seq_len=MAX_SEQ_LEN)

    # 3. Setup Optimizer
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=LEARNING_RATE, weight_decay=0.01)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Optimizer] Successfully restored optimizer state.")
        except Exception:
            print("[Optimizer] Re-initialized optimizer state for true ternary training.")

    # 4. Graceful Exit Handler
    start_step = ckpt.get("global_step", 0)
    current_step = start_step
    training_start_time = time.time()
    should_exit = False

    def handle_signal(sig, frame):
        nonlocal should_exit
        print("\n[Exit Signal] Shutdown requested. Calcifying weights atomically before exit...")
        should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n" + "─" * 70)
    print("  TRAINING ACTIVE — 1-HOUR TIMED PASS")
    print("─" * 70)

    step_loss_history = []
    log_every = 25
    save_every = 100

    while not should_exit:
        elapsed = time.time() - training_start_time
        if elapsed >= TARGET_DURATION_SECONDS:
            print(f"\n[Milestone] 1-Hour Target Reached ({elapsed:.1f}s elapsed). Finalizing session.")
            break

        current_step += 1
        optimizer.zero_grad()

        # Batch
        batch = packer.get_batch(BATCH_SIZE, device=device)
        tokens = batch["text"]

        # Embed & Target
        with torch.no_grad():
            clean_latent = nexus.core.text_stack.integrate_input(tokens)

        # Diffusion Step
        total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
            nexus, clean_latent, max_T=3,
            ce_target=tokens, ce_weight=0.3,
            use_min_snr=False
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
        optimizer.step()

        # Ambient Consolidation Tick
        nexus.ambient_field.tick_consolidation()

        loss_val = total_loss.item()
        step_loss_history.append(loss_val)

        # Logging
        if current_step % log_every == 0:
            avg_loss = sum(step_loss_history[-log_every:]) / log_every
            ppl = math.exp(min(20, avg_loss))
            remaining = max(0, TARGET_DURATION_SECONDS - elapsed)
            rem_min = remaining / 60.0
            print(f"  Step {current_step:,} | Loss: {avg_loss:.4f} | PPL: {ppl:8.2f} | "
                  f"Elapsed: {elapsed/60:.1f}m / 60.0m ({rem_min:.1f}m left)")

        # Periodic Safe Save
        if current_step % save_every == 0:
            save_dict = {
                "model_state_dict": nexus.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": current_step,
                "config": config,
                "ternary_spec": "BitNet_1.58b_strict_three_state"
            }
            PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path, create_backup=False)

        if "mps" in str(device) and current_step % 20 == 0:
            torch.mps.empty_cache()

    # Final Save
    total_elapsed = time.time() - training_start_time
    print(f"\n[Complete] Total Elapsed Time: {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"[Complete] Final Global Step: {current_step:,} (Trained {current_step - start_step:,} new steps)")

    final_save_dict = {
        "model_state_dict": nexus.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": current_step,
        "config": config,
        "ternary_spec": "BitNet_1.58b_strict_three_state",
        "training_duration_seconds": total_elapsed
    }
    
    # Save both main checkpoint and dedicated true ternary checkpoint
    PCEINexus.save_safe_checkpoint(nexus, final_save_dict, ckpt_path, create_backup=True)
    out_true_ternary = PROJECT_ROOT / "checkpoints" / "pcei_100m_true_ternary.pt"
    PCEINexus.save_safe_checkpoint(nexus, final_save_dict, out_true_ternary, create_backup=False)
    print(f"[Saved] {ckpt_path.name} and {out_true_ternary.name} updated successfully.")

if __name__ == "__main__":
    main()
