import torch
import torch.nn as nn
import math
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.engine.diffusion_utils import diffusion_training_step
from scripts.train import PCEITokenizer

def resonance_chat(args):
    print("═" * 60)
    print("  PCEI Resonance Chat — Real-Time Emergence")
    print("  Mode: Active Learning (Every interaction is a training step)")
    print("═" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = PCEITokenizer()
    
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        checkpoint_path, 2, 1, 1
    )
    print(f"[Topology] Detected: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    
    # Load Micro-Nexus
    nexus = PCEINexus(
        vocab_size=args.vocab_size,
        device=device,
        d_model_text=d_text,
        d_model_img=d_img,
        d_model_audio=d_audio,
        layers_text=layers_text,
        layers_img=layers_img,
        layers_audio=layers_audio
    )
    if checkpoint_path.exists():
        print(f"[Loading] {checkpoint_path.name}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # --- WEIGHT SANITIZER ---
            # Scans every parameter for NaNs hiding from previous crashes
            with torch.no_grad():
                nan_count = 0
                # Scrub Parameters
                for p in nexus.parameters():
                    mask = torch.isnan(p) | torch.isinf(p)
                    if mask.any():
                        nan_count += mask.sum().item()
                        p.data[mask] = 0.0
                
                # Scrub Buffers (Ambient Field, etc.)
                for b in nexus.buffers():
                    mask = torch.isnan(b) | torch.isinf(b)
                    if mask.any():
                        nan_count += mask.sum().item()
                        b.data[mask] = 0.0

                if nan_count > 0:
                    print(f"[Sanitizer] Cleaned {nan_count} corrupted elements from foundation.")
            
            print("[Success] Foundation loaded and sanitized.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load checkpoint: {e}")
            print("Aborting to protect existing data.")
            sys.exit(1)
    else:
        print("[Warning] No checkpoint found. Starting from blank noise.")
    
    nexus.train() # Keep in train mode for active learning
    
    # --- WEIGHT SANITIZER & DEFIBRILLATOR ---
    with torch.no_grad():
        nan_count = 0
        revived_count = 0
        for name, p in nexus.named_parameters():
            # 1. Scrub NaNs/Infs
            mask = torch.isnan(p) | torch.isinf(p)
            if mask.any():
                nan_count += mask.sum().item()
                p.data[mask] = 0.0
            
            # 2. Defibrillator: Detect Dead Layers (All Zero)
            if "weight" in name and p.numel() > 10:
                zero_ratio = (p.data == 0).float().mean().item()
                if zero_ratio > 0.90:
                    revived_count += 1
                    # Spec-Correct Initialization for revived layers
                    if "dt_proj" in name:
                        # Small uniform for dt_proj weights
                        dt_init_std = 0.01 # Safe default
                        nn.init.uniform_(p.data, -dt_init_std, dt_init_std)
                    elif "A_log" in name:
                        # Log range for A
                        A = torch.arange(1, p.size(1) + 1).float()
                        p.data.copy_(torch.log(A.repeat(p.size(0), 1)))
                    elif "D" in name:
                        nn.init.ones_(p.data)
                    elif p.dim() >= 2:
                        nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                    else:
                        nn.init.normal_(p.data, std=0.02)
                    
            if "bias" in name and "dt_proj" in name:
                # Special bias init for Mamba dt
                dt_init = torch.exp(torch.rand_like(p.data) * -4.0 - 2.0) # 0.001 to 0.1
                inv_softplus = lambda x: torch.log(torch.expm1(x))
                p.data.copy_(inv_softplus(dt_init))
        
        if nan_count > 0:
            print(f"[Sanitizer] Cleaned {nan_count} corrupted weights.")
        if revived_count > 0:
            print(f"[Defibrillator] Revived {revived_count} collapsed layers.")

    optimizer = torch.optim.AdamW(nexus.parameters(), lr=5e-5)
    
    print("\n[READY] The model is currently pure noise. Talk to it.")
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            # 1. WITNESS PHASE (Diffusion Denoising Training)
            # ----------------------------------------
            with torch.no_grad():
                # Encode and get clean latents
                prompt_ids = tokenizer.enc.encode(user_input, allowed_special={"<|endoftext|>"})
                prompt_ids = [tid if tid != 0 else (tokenizer.vocab_size - 1) for tid in prompt_ids]
                input_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)
                clean_latent = nexus.core.text_stack.integrate_input(input_tokens)
            
            # Diffusion training: sample random timestep, add noise, denoise
            # CE auxiliary loss keeps the lm_head projection sharp for generation
            optimizer.zero_grad()
            
            total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                nexus, clean_latent, max_T=args.depth,
                ce_target=input_tokens, ce_weight=0.3
            )
            
            total_loss.backward()
            
            # --- SCORCH EARTH GRADIENT SANITIZER ---
            for p in nexus.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0)
                    
            # --- DYNAMIC PLASTICITY TICK ---
            nexus.adjust_plasticity(optimizer)
            
            torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
            optimizer.step()
            
            clean_loss = float(torch.nan_to_num(torch.tensor(mse_val), nan=0.0))
            active_bonds = len(nexus.core.broker.active_bonds)
            print(f"[Learning] Physics Loss (MSE): {clean_loss:.6f} | Active Bonds: {active_bonds} | t={sampled_t}")

            # 2. RESONANCE PHASE (Full-Spec Parallel Diffusion)
            # ----------------------------------------
            nexus.eval()
            with torch.no_grad():
                # 1. Integrate prompt to latents
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
                prompt_latent = nexus.core.text_stack.integrate_input(prompt_tensor)
                
                # 2. Use DreamSeeder to generate field-informed noise
                # The intent_vector is the mean of your prompt
                # Calculate actual mean confidence
                conf_vals = [m.slow_field_confidence.item() for m in nexus.core.modules() if hasattr(m, 'slow_field_confidence')]
                actual_mean_conf = sum(conf_vals) / max(len(conf_vals), 1) if conf_vals else 0.5

                intent = prompt_latent.mean(dim=1)
                dream_seeds = nexus.core.dream_seeder(
                    intent_vector=intent,
                    batch_size=1,
                    seq_lens={'text': 32}, # 32-token response block
                    slow_state=nexus.ambient_field.slow_state,
                    mean_confidence=actual_mean_conf
                )
                resp_noise = dream_seeds['text']
                
                # 3. Concatenate: [Prompt (Stable) | Response (Noisy)]
                current_latent = torch.cat([prompt_latent, resp_noise], dim=1)
                
                # 4. Iterative Denoising Loop (True-to-Spec Diffusion)
                nexus.response_gate.reset()
                thought_settled = False
                for t in reversed(range(args.depth)): 
                    # We call the core with the latent and the AmbientField state
                    device_type = 'cuda' if 'cuda' in str(device) else ('mps' if 'mps' in str(device) else 'cpu')
                    amp_dtype = torch.float16 if device_type == 'mps' else (torch.bfloat16 if device_type == 'cuda' else torch.float32)
                    
                    with torch.autocast(device_type='cuda' if device_type == 'cuda' else 'cpu', dtype=amp_dtype, enabled=(device_type == 'cuda')):
                        _, denoised, _, _, _, _ = nexus.core(
                            text_latent=current_latent,
                            diffusion_timesteps=1, # Single step iteration
                            slow_state=nexus.ambient_field.slow_state,
                            mean_confidence=actual_mean_conf
                        )
                    
                    # Anchor the prompt, update only the response part
                    current_latent[:, prompt_latent.size(1):, :] = torch.clamp(denoised[:, prompt_latent.size(1):, :], -8.0, 8.0)
                    
                    # PCEI Feature: Coherence Gating
                    if nexus.response_gate.check_settlement(current_latent[:, prompt_latent.size(1):, :]):
                        thought_settled = True
                        break
                
                # 5. Decode only if the thought has settled
                if thought_settled:
                    resp_latent = current_latent[:, prompt_latent.size(1):, :]
                    logits = nexus.core.text_stack.decode_output(resp_latent)
                    
                    # Sampling with Temperature
                    probs = torch.softmax(logits[0] / 0.8, dim=-1)
                    generated_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()
                else:
                    generated_ids = []

            nexus.core.dream_seeder.clear()
            nexus.train()
            
            # Filter out pads and detokenize
            decode_ids = [tid for tid in generated_ids if tid != tokenizer.pad_id]
            decode_ids = [0 if tid == (tokenizer.vocab_size - 1) else tid for tid in decode_ids]
            response_text = tokenizer.enc.decode(decode_ids)
            
            if response_text.strip():
                print(f"MODEL: {response_text.strip()}")
            else:
                print("MODEL: [silence]")

            
            # 3. CONSOLIDATION PHASE (Physics Tick)
            # ----------------------------------------
            # Use the final denoised latent from the response phase for field consolidation
            # current_latent holds the last denoised response latent
            nexus.ambient_field.tick(current_latent, clean_loss, detectors=nexus.core.detectors)
            nexus.core.broker.measure_entanglement(nexus.core.detectors)
            nexus.core.broker.tick_decay()
            nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
            nexus.core.shifter.refresh_registry()
            
            # --- ATOMIC SAVE ---
            backup_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_BACKUP.pt")
            import shutil
            if checkpoint_path.exists():
                shutil.copy2(checkpoint_path, backup_path)
            
            torch.save({
                'model_state_dict': nexus.state_dict(),
                'global_step': 0, # Step is arbitrary in resonance chat
                'config': checkpoint.get('config', {}) if 'checkpoint' in locals() else {}
            }, str(checkpoint_path) + '.tmp')
            import os
            os.replace(str(checkpoint_path) + '.tmp', str(checkpoint_path))
            print("[Saved] Resonance state consolidated.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_micro.pt")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=50257)
    args = parser.parse_args()
    resonance_chat(args)
