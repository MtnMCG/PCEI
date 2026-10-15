import torch
import torch.nn as nn
import math
import argparse
import sys
import os
import time
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pcei import PCEINexus
from src.engine.diffusion_utils import diffusion_training_step
from scripts.train import PCEITokenizer
from dashboard.telemetry import PCEITelemetry

DIRECTOR_FILE = Path(__file__).resolve().parent.parent / "checkpoints" / "director_prompt.txt"

def query_ollama(model_name, messages):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"[Ollama Error] Failed to reach Ollama: {e}")
    return "Hello. I am having trouble connecting to my local systems."

def ollama_trainer(args):
    print("═" * 60)
    print("  PCEI Ollama Trainer — Director Mode")
    print(f"  Ollama Model: {args.ollama_model}")
    print(f"  Learning Rate: {args.lr}")
    print("═" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = PCEITokenizer()
    
    telem = PCEITelemetry.from_env()
    telem.install_graceful_shutdown()
    
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        checkpoint_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    
    # Load Nexus
    nexus = PCEINexus(config=config, device=device)
    checkpoint = None
    if checkpoint_path.exists():
        print(f"[Loading] {checkpoint_path.name}...", flush=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nexus.load_ambient_state(checkpoint)
            print("[Success] Foundation loaded.", flush=True)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load checkpoint: {e}", flush=True)
            sys.exit(1)
    else:
        print("[Warning] No checkpoint found. Starting from blank weights.", flush=True)

    nexus.train()
    optimizer = torch.optim.AdamW(nexus.parameters(), lr=args.lr)
    global_step = checkpoint.get('global_step', 0) if checkpoint else 0

    # Initialize Ollama Chat History
    system_prompt = """You are an advanced AI model (Ollama) playing the role of a human user. 
You are currently interacting with "PCEI", an experimental 100M-parameter AI model that is actively learning from this conversation. 
Your goal is to converse with PCEI naturally, explore diverse topics, and help it learn how to structure English sentences.
PCEI is currently learning syntax, so it might occasionally stutter or output strange tokens (like "ASSASS::"). If it does, patiently continue the conversation or gently guide it back on track.
You act as its mentor by providing high-quality, perfectly structured responses for it to learn from.
Keep your messages concise and conversational. Do NOT prefix your messages with 'USER:'."""

    ollama_messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("[Ready] Autonomous loop starting. Awaiting Director instructions...", flush=True)

    while True:
        if telem.shutdown_requested:
            break

        # --- 1. DIRECTOR INSTRUCTION CHECK ---
        if DIRECTOR_FILE.exists():
            try:
                director_cmd = DIRECTOR_FILE.read_text().strip()
                DIRECTOR_FILE.unlink()
                if director_cmd:
                    print(f"\n[bold yellow]🎬 DIRECTOR:[/bold yellow] {director_cmd}", flush=True)
                    # Add as a system instruction to guide Ollama's next move
                    ollama_messages.append({"role": "system", "content": f"DIRECTOR INSTRUCTION for your next message: {director_cmd}"})
            except Exception:
                pass

        # --- 2. OLLAMA (USER) PHASE ---
        print("\n[Ollama Thinking...]", flush=True)
        ollama_response = query_ollama(args.ollama_model, ollama_messages)
        ollama_messages.append({"role": "assistant", "content": ollama_response})
        
        # We display it as USER because Ollama is playing the role of the User talking to PCEI
        print(f"\nUSER: {ollama_response}", flush=True)

        # --- 3. PCEI (ASSISTANT) TRAINING PHASE ---
        formatted_prompt = f"USER: {ollama_response}\nASSISTANT:"
        
        with torch.no_grad():
            prompt_ids = tokenizer.enc.encode(formatted_prompt, allowed_special={"<|endoftext|>"})
            prompt_ids = [tid if tid != 0 else (tokenizer.vocab_size - 1) for tid in prompt_ids]
            input_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            clean_latent = nexus.core.text_stack.integrate_input(input_tokens)

        optimizer.zero_grad()
        total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
            nexus, clean_latent, max_T=12,
            ce_target=input_tokens, ce_weight=0.3
        )
        total_loss.backward()

        nexus.adjust_plasticity(optimizer)
        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
        optimizer.step()
        global_step += 1

        if getattr(nexus, '_structure_changed', False):
            optimizer = torch.optim.AdamW(nexus.parameters(), lr=args.lr)
            nexus._structure_changed = False
            print("[Neurogenesis] Optimizer rebuilt.", flush=True)

        clean_loss = mse_val
        active_bonds = len(nexus.core.broker.active_bonds) if hasattr(nexus.core.broker, 'active_bonds') else 0
        print(f"[Step {global_step}] Loss: {clean_loss:.6f} | Active Bonds: {active_bonds} | t={sampled_t}", flush=True)

        # --- 4. PCEI (ASSISTANT) GENERATION PHASE ---
        nexus.eval()
        print("MODEL: ", end="", flush=True)

        generated_ids = []
        with torch.inference_mode():
            current_latent = clean_latent
            conf_vals = [m.slow_field_confidence.item() for m in nexus.core.modules() if hasattr(m, 'slow_field_confidence')]
            actual_mean_conf = sum(conf_vals) / max(len(conf_vals), 1) if conf_vals else 0.5
            
            MAX_NEW_TOKENS = 64
            device_type = 'cuda' if 'cuda' in str(device) else ('mps' if 'mps' in str(device) else 'cpu')
            amp_dtype = torch.float16 if device_type == 'mps' else (torch.bfloat16 if device_type == 'cuda' else torch.float32)
            ssm_cache = {}

            # Initial inference pass to get the latent for generation
            with torch.autocast(device_type='cuda' if device_type == 'cuda' else 'cpu', dtype=amp_dtype, enabled=(device_type == 'cuda')):
                nexus.response_gate.reset()
                _, denoised, _, _, _, _ = nexus.core(
                    text_latent=current_latent,
                    diffusion_timesteps=1, override_timestep=0,
                    slow_state=nexus.ambient_field.slow_state, mean_confidence=actual_mean_conf, cache=ssm_cache
                )
                last_token_latent = denoised[:, -1:, :]

            start_t = time.time()
            for step in range(MAX_NEW_TOKENS):
                logits = nexus.core.text_stack.decode_output(last_token_latent)
                probs = torch.softmax(logits[0, 0, :] / 0.7, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == (tokenizer.vocab_size - 1) or next_token_id == tokenizer.pad_id:
                    break

                generated_ids.append(next_token_id)
                word = tokenizer.enc.decode([next_token_id])
                print(word, end="", flush=True)

                next_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                current_latent = nexus.core.text_stack.integrate_input(next_tensor)

                with torch.autocast(device_type='cuda' if device_type == 'cuda' else 'cpu', dtype=amp_dtype, enabled=(device_type == 'cuda')):
                    _, denoised, _, _, _, _ = nexus.core(
                        text_latent=current_latent,
                        diffusion_timesteps=1, override_timestep=0,
                        slow_state=nexus.ambient_field.slow_state, mean_confidence=actual_mean_conf, cache=ssm_cache
                    )
                last_token_latent = denoised[:, -1:, :]
                
            elapsed = time.time() - start_t
            tps = len(generated_ids) / elapsed if elapsed > 0 else 0

        print()  # Newline after generation
        nexus.train()
        
        telem.emit(
            step=global_step,
            loss=clean_loss,
            mean_confidence=actual_mean_conf,
            lr=args.lr,
            tokens_per_sec=tps,
            active_bonds=active_bonds
        )

        pcei_response = tokenizer.enc.decode(generated_ids)
        # Add PCEI response to Ollama's context so Ollama knows what PCEI said
        ollama_messages.append({"role": "user", "content": pcei_response})
        
        # Keep context window manageable (System prompt + last 20 messages)
        if len(ollama_messages) > 21:
            ollama_messages = [ollama_messages[0]] + ollama_messages[-20:]

        # --- 5. CONSOLIDATION & SAVE ---
        nexus.ambient_field.tick(current_latent, clean_loss, detectors=nexus.core.detectors)
        nexus.core.broker.measure_entanglement(nexus.core.detectors)
        nexus.core.broker.tick_decay()
        nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
        nexus.core.shifter.refresh_registry()

        save_dict = {
            'model_state_dict': nexus.state_dict(),
            'global_step': global_step,
        }
        if PCEINexus.save_safe_checkpoint(nexus, save_dict, checkpoint_path, create_backup=True):
            print("[Saved] Resonance state consolidated.", flush=True)
            
        time.sleep(1) # Small pause so the user can read

    telem.close()
    print("\n[System] Ollama Trainer stopped cleanly.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt")
    parser.add_argument("--ollama-model", type=str, default="MtnMCG/Council-Ultima:latest")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    # Optional Rich formatting for terminal
    try:
        from rich import print
    except ImportError:
        pass
        
    ollama_trainer(args)
