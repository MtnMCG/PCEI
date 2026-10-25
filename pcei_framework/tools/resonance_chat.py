import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.engine.diffusion_utils import diffusion_training_step, ddpm_reverse_step
from src.tokenizer import PCEITokenizer

def get_parasitic_junk_tokens(tokenizer, device="cpu"):
    bad_ids = set(range(94, 256))
    junk_patterns = [
        'reportprint', 'rawdownload', 'RandomRedditor', 'TheNitrome',
        'externalTo', 'oreAndOnline', 'quickShip', 'StreamerBot',
        'ActionCode', 'RecipeTag', 'SubmitButton', 'InstoreAndOnline',
        'DiaryInfo', 'ÃÂ', 'DeliveryDate', 'Nitrome', 'ThumbnailImage',
        'oreAnd', 'Diary', 'Recipe', 'Recipes', '="', '=\"', '</', '/>', '><'
    ]
    for i in range(tokenizer.vocab_size):
        try:
            s = tokenizer.enc.decode([i])
            if any(p in s for p in junk_patterns) or any(ord(c) > 1000 for c in s) or (len(s) == 1 and ord(s) < 32 and s not in '\n\t '):
                bad_ids.add(i)
        except Exception:
            bad_ids.add(i)
    return torch.tensor(list(bad_ids), device=device)

def get_syntax_token_ids(tokenizer):
    syntax_words = [
        ' the', ' of', ' and', ' to', ' in', ' a', ' is', ' that', ' for', ' on',
        ' with', ' as', ' by', ' at', ' from', ' it', ' an', ' be', ' was', ' are',
        '.', ',', ' ', '\n', '-', ':', '(', ')', '"', "'", '!', '?', ';'
    ]
    syntax_ids = set()
    for w in syntax_words:
        for t in tokenizer.encode(w, pad=False).tolist():
            syntax_ids.add(t)
    return syntax_ids

def get_banned_ngrams(generated_ids, n=3):
    if len(generated_ids) < n - 1 or n <= 1:
        return []
    prefix = tuple(generated_ids[-(n - 1):])
    banned = []
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i + n - 1]) == prefix:
            banned.append(generated_ids[i + n - 1])
    return banned

def resonance_chat(args):
    print("═" * 60)
    print("  PCEI Resonance Chat — Real-Time Emergence")
    print("  Mode: Active Learning (Every interaction is a training step)")
    print("═" * 60)

    if getattr(args, 'device', None):
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = PCEITokenizer()
    bad_tensor = get_parasitic_junk_tokens(tokenizer, device=device)
    syntax_ids = get_syntax_token_ids(tokenizer)
    
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    
    ckpt_arg = args.ckpt if getattr(args, 'ckpt', None) else args.checkpoint_name
    checkpoint_path = Path(ckpt_arg)
    if not checkpoint_path.exists():
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / ckpt_arg

    if not checkpoint_path.exists():
        print(f"\n⚠️  [WARNING] Requested checkpoint '{ckpt_arg}' was not found on your local disk!")
        print(f"    (Note: If syncing from the cloud, the file is still transferring).")
        # Fallback candidate search
        ckpt_dir = Path(__file__).parent.parent / "checkpoints"
        candidates = [
            "pcei_100m_conversational.pt",
            "pcei_100m_sft.pt",
            "pcei_100m.pt",
        ]
        for cand in candidates:
            cand_p = ckpt_dir / cand
            if cand_p.exists():
                checkpoint_path = cand_p
                print(f"[Fallback] Falling back to available checkpoint: {cand}\n")
                break

    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        checkpoint_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    print(f"[Topology] Detected: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio

    # Load Nexus
    nexus = PCEINexus(config=config, device=device)
    checkpoint = None
    if checkpoint_path.exists():
        print(f"[Loading] {checkpoint_path.name}...", flush=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nexus.load_ambient_state(checkpoint)
            
            # Weight sanitizer — scrub NaNs
            with torch.no_grad():
                nan_count = 0
                for p in nexus.parameters():
                    mask = torch.isnan(p) | torch.isinf(p)
                    if mask.any():
                        nan_count += mask.sum().item()
                        p.data[mask] = 0.0
                for b in nexus.buffers():
                    mask = torch.isnan(b) | torch.isinf(b)
                    if mask.any():
                        b.data[mask] = 0.0
                if nan_count > 0:
                    print(f"[Sanitizer] Cleaned {nan_count} corrupted elements.", flush=True)

            print("[AmbientField] Persistent consciousness loaded from checkpoint.", flush=True)
            print("[Success] Foundation loaded and sanitized.", flush=True)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load checkpoint: {e}", flush=True)
            sys.exit(1)
    else:
        print("[Warning] No checkpoint found. Starting from blank weights.", flush=True)

    if getattr(args, 'metabolic_flush', False):
        print("[System] Metabolic Flush: Resetting slow-field confidence...", flush=True)
        with torch.no_grad():
            from src.modules.plasticity import PlasticLayer
            for m in nexus.core.modules():
                if isinstance(m, PlasticLayer):
                    m.slow_field_confidence.zero_()

    nexus.train()



    optimizer = torch.optim.AdamW(nexus.parameters(), lr=5e-5)
    global_step = 0
    if checkpoint is not None and isinstance(checkpoint, dict):
        global_step = checkpoint.get('global_step', 0)

    # Block non-interactive mode unless explicitly in api_mode
    if not sys.stdin.isatty() and not getattr(args, 'api_mode', False):
        print("[Notice] Resonance Chat requires a terminal session. Exiting cleanly.", flush=True)
        return

    inference_only = getattr(args, 'inference_only', True)
    if getattr(args, 'active_learning', False):
        inference_only = False

    if inference_only:
        print("[Mode] Inference-only (no training, fast responses)", flush=True)
    else:
        print("[Mode] Active Learning (trains & consolidates on every message)", flush=True)
    print(f"[Sampling] Temperature: {args.temperature} | Top-p: {args.top_p}", flush=True)
    print("[Context] Multi-turn dialogue memory active (/clear, /history, /bonds, /status)", flush=True)

    conversation_history = []  # List of (user_text, asst_text)
    MAX_PROMPT_TOKENS = getattr(args, 'max_context_tokens', 1800)
    turns_since_save = 0

    # ─── MAIN CHAT LOOP ──────────────────────────────────────────────────────
    while True:

        # --- INPUT PHASE ---
        try:
            if getattr(args, 'api_mode', False):
                user_input = input().strip()
            else:
                user_input = input("\nYOU: ").strip()

            if user_input.lower() in ["exit", "quit", "/exit", "/quit", ":q", "q", "bye", "/bye", "/stop"]:
                break
            if not user_input or not user_input.strip():
                continue

            # Special commands
            if user_input.lower() in ["/clear", "/reset"]:
                conversation_history.clear()
                print("[Memory] Conversation history cleared.", flush=True)
                continue
            elif user_input.lower() == "/history":
                print(f"\n[History] {len(conversation_history)} turn(s) in memory:")
                for idx, (u, a) in enumerate(conversation_history):
                    print(f"  [{idx+1}] YOU: {u}")
                    print(f"      MODEL: {a}")
                continue
            elif user_input.lower() == "/bonds":
                bonds = getattr(nexus.core.broker, 'active_bonds', {})
                print(f"[Broker] Active Topological Bonds: {len(bonds)}")
                sorted_b = sorted(bonds.items(), key=lambda x: x[1], reverse=True)[:8]
                for (k1, k2), s in sorted_b:
                    print(f"  {k1:12s} <---> {k2:12s} : {s:.4f}")
                continue
            elif user_input.lower() == "/status":
                bonds = getattr(nexus.core.broker, 'active_bonds', {})
                print(f"[Status] Global Step: {global_step} | Active Bonds: {len(bonds)} | Memory Turns: {len(conversation_history)}")
                continue
            elif user_input.lower() == "/save":
                save_dict = {
                    'model_state_dict': nexus.state_dict(),
                    'global_step': global_step,
                    'config': checkpoint.get('config', {}) if checkpoint is not None else {}
                }
                if PCEINexus.save_safe_checkpoint(nexus, save_dict, checkpoint_path, create_backup=True):
                    print("[Saved] Checkpoint saved successfully.", flush=True)
                continue

        except (EOFError, KeyboardInterrupt):
            break

        # --- BUILD MULTI-TURN CONTEXT WITH SLIDING WINDOW ---
        turns_to_use = [] if getattr(args, 'single_turn', False) else list(conversation_history)
        while True:
            context_blocks = []
            for u_past, a_past in turns_to_use:
                context_blocks.append(f"USER: {u_past}\nASSISTANT: {a_past}<|endoftext|>")
            context_blocks.append(f"USER: {user_input}\nASSISTANT:")
            formatted_prompt = "\n".join(context_blocks)

            prompt_ids = tokenizer.enc.encode(formatted_prompt, allowed_special={"<|endoftext|>"})

            if len(prompt_ids) <= MAX_PROMPT_TOKENS or not turns_to_use:
                break
            # Pop oldest turn to fit in budget
            turns_to_use.pop(0)

        # --- WITNESS PHASE (Diffusion Denoising Training) ---
        try:
            with torch.no_grad():
                input_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)
                clean_latent = nexus.core.text_stack.integrate_input(input_tokens)

            if not inference_only:
                optimizer.zero_grad()
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, clean_latent, max_T=args.depth,
                    ce_target=input_tokens, ce_weight=0.3
                )
                total_loss.backward()

                nexus.adjust_plasticity(optimizer)
                torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=0.5)
                optimizer.step()
                global_step += 1

                if getattr(nexus, '_structure_changed', False):
                    optimizer = torch.optim.AdamW(nexus.parameters(), lr=5e-5)
                    nexus._structure_changed = False
                    print("[Neurogenesis] Optimizer rebuilt.", flush=True)

                clean_loss = mse_val
                active_bonds = len(nexus.core.broker.active_bonds) if hasattr(nexus.core.broker, 'active_bonds') else 0
                print(f"[Step {global_step}] Loss: {clean_loss:.6f} | Active Bonds: {active_bonds} | t={sampled_t}", flush=True)
            else:
                clean_loss = 0.0
                active_bonds = 0

            # --- RESONANCE PHASE (Autoregressive Generation) ---
            nexus.eval()
            if getattr(args, 'api_mode', False):
                print("\n__START_MODEL__", flush=True)
            else:
                print("MODEL: ", end="", flush=True)

            generated_ids = []

            with torch.inference_mode():
                curr_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)
                MAX_NEW_TOKENS = 248

                nexus.response_gate.reset()

                try:
                    for step in range(MAX_NEW_TOKENS):
                        logits, *_ = nexus.core(
                            text_input=curr_tokens,
                            diffusion_timesteps=1,
                            override_timestep=0,
                            slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
                        )
                        # Cast to float32 for all sampling math — bfloat16 softmax/cumsum
                        # has too little precision for the tight logit gaps at this training stage
                        logits_last = logits[:, -1, :].squeeze(0).clone().float()

                        # 1. Mask out junk/parasitic tokens (if enabled)
                        if getattr(args, 'mask_junk', True):
                            logits_last[bad_tensor] = -1e4

                        # 2. Hard n-gram repetition blocking (prevents multi-token limit cycles)
                        no_repeat_n = getattr(args, 'no_repeat_ngram', 3)
                        if no_repeat_n > 0 and len(generated_ids) >= no_repeat_n - 1:
                            banned_ng = get_banned_ngrams(generated_ids, n=no_repeat_n)
                            if banned_ng:
                                logits_last[torch.tensor(banned_ng, device=device)] = -1e4

                        # 3. Non-syntax consecutive repeat ban (x_t != x_{t-1})
                        if len(generated_ids) > 0:
                            last_tok = generated_ids[-1]
                            if last_tok not in syntax_ids:
                                logits_last[last_tok] = -1e4

                        # 4. Additive frequency penalty on tokens appearing 2+ times
                        freq_pen = getattr(args, 'freq_penalty', 4.0)
                        if freq_pen > 0 and len(generated_ids) > 0:
                            recent = generated_ids[-40:]
                            counts = torch.bincount(torch.tensor(recent, device=device), minlength=tokenizer.vocab_size)
                            repeat_mask = (counts >= 2)
                            for s_id in syntax_ids:
                                if s_id < len(repeat_mask):
                                    repeat_mask[s_id] = False
                            logits_last[repeat_mask] -= freq_pen

                        # 5. Multiplicative repetition penalty on recent tokens
                        rep_pen = getattr(args, 'repetition_penalty', 1.15)
                        if rep_pen > 1.0 and len(generated_ids) > 0:
                            recent_unique = set(generated_ids[-40:])
                            for t in recent_unique:
                                if t not in syntax_ids:
                                    if logits_last[t] > 0:
                                        logits_last[t] /= rep_pen
                                    else:
                                        logits_last[t] *= rep_pen

                        # Early EOS Suppression (Forces model to formulate complete thoughts instead of bailing on 1 token)
                        min_toks = getattr(args, 'min_tokens', 4)
                        if step < min_toks:
                            logits_last[tokenizer.eos_id] = -1e4

                        scaled = logits_last / max(1e-4, args.temperature)

                        # Top-p nucleus filtering
                        probs = F.softmax(scaled, dim=-1)
                        sprobs, sidx = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sprobs, dim=-1)
                        remove_mask = cum - sprobs > args.top_p
                        probs[sidx[remove_mask]] = 0.0
                        probs_sum = probs.sum()
                        if probs_sum > 0:
                            probs /= probs_sum
                            next_token_id = torch.multinomial(probs, num_samples=1).item()
                        else:
                            next_token_id = sidx[0].item()

                        if next_token_id == tokenizer.eos_id:
                            if step >= min_toks:
                                break
                            else:
                                continue

                        generated_ids.append(next_token_id)
                        word = tokenizer.enc.decode([next_token_id])

                        full_gen = tokenizer.enc.decode(generated_ids)
                        if any(stop in full_gen for stop in ["\nUSER:", "\nHuman:", "<|endoftext|>", "USER:", "\nUSER"]):
                            break

                        print(word, end="", flush=True)
                        curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
                except KeyboardInterrupt:
                    print("\n[Generation interrupted by user]", flush=True)

            if getattr(args, 'api_mode', False):
                print("__END_MODEL__", flush=True)
            else:
                print()  # Newline after generation

            # Extract clean assistant generation and append to multi-turn history
            full_gen_clean = tokenizer.enc.decode(generated_ids)
            for stop_seq in ["\nUSER:", "\nHuman:", "<|endoftext|>", "USER:", "\nUSER"]:
                if stop_seq in full_gen_clean:
                    full_gen_clean = full_gen_clean.split(stop_seq)[0]
            clean_response = full_gen_clean.strip()
            if not clean_response and grounded_fact:
                clean_response = grounded_fact
            if clean_response:
                conversation_history.append((user_input, clean_response))

            nexus.train()

            # --- CONSOLIDATION PHASE (Physics Tick) — skip in inference-only ---
            if not inference_only:
                nexus.ambient_field.tick(clean_latent, clean_loss, detectors=nexus.core.detectors)
                nexus.core.broker.measure_entanglement(nexus.core.detectors)
                nexus.core.broker.tick_decay()
                nexus.core.shifter.apply_forces(nexus.core.broker.active_bonds)
                nexus.core.shifter.refresh_registry()

                turns_since_save += 1
                if turns_since_save >= 10:
                    save_dict = {
                        'model_state_dict': nexus.state_dict(),
                        'global_step': global_step,
                        'config': checkpoint.get('config', {}) if checkpoint is not None else {}
                    }
                    if PCEINexus.save_safe_checkpoint(nexus, save_dict, checkpoint_path, create_backup=True):
                        print("[Saved] Periodic resonance state consolidated.", flush=True)
                    turns_since_save = 0

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error] {e}", flush=True)

    # Save once on exit if active learning was enabled
    if not inference_only and turns_since_save > 0:
        print("\n[Saving] Consolidating learned resonance before exit...", flush=True)
        save_dict = {
            'model_state_dict': nexus.state_dict(),
            'global_step': global_step,
            'config': checkpoint.get('config', {}) if checkpoint is not None else {}
        }
        PCEINexus.save_safe_checkpoint(nexus, save_dict, checkpoint_path, create_backup=True)
        print("[Saved] Final resonance state saved.", flush=True)

    print("\n[Resonance Chat] Session closed. Farewell!", flush=True)


if __name__ == "__main__":
    import subprocess, os, platform, sys
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpt", "--checkpoint", type=str, default=None, help="Path to checkpoint file")
        parser.add_argument("--checkpoint-name", type=str, default="pcei_100m_conversational.pt",
                            help="Default checkpoint to load (default: pcei_100m_conversational.pt)")
        parser.add_argument("--device", type=str, default=None, help="Device to run on (mps, cuda, cpu)")
        parser.add_argument("--depth", type=int, default=12)
        parser.add_argument("--vocab-size", type=int, default=50257)
        parser.add_argument("--min-tokens", type=int, default=4,
                            help="Minimum tokens to generate before allowing EOS (default: 4)")
        parser.add_argument("--single-turn", action="store_true", default=False,
                            help="Disable multi-turn conversation history; each message is treated independently")
        parser.add_argument("--metabolic-flush", action="store_true", default=False,
                            help="Reset slow-field confidence to zero before conversation")
        parser.add_argument("--inference-only", action="store_true", default=True,
                            help="Skip training backward pass — fast inference, no active learning")
        parser.add_argument("--active-learning", action="store_true", default=False,
                            help="Enable continuous online learning and physics consolidation on each message")
        parser.add_argument("--temperature", type=float, default=0.7,
                            help="Sampling temperature (default: 0.7)")
        parser.add_argument("--top-p", type=float, default=0.90,
                            help="Top-p nucleus sampling threshold (default: 0.90)")
        parser.add_argument("--repetition-penalty", type=float, default=1.15,
                            help="Multiplicative repetition penalty on recent 40 tokens (default: 1.15)")
        parser.add_argument("--freq-penalty", type=float, default=0.0,
                            help="Additive frequency penalty on repeated content tokens (default: 0.0)")
        parser.add_argument("--no-repeat-ngram", type=int, default=0,
                            help="Bans completion of identical N-grams in generation (default: 0, disabled)")
        parser.add_argument("--mask-junk", action="store_true", default=True,
                            help="Mask parasitic/scraping/byte tokens from vocabulary logits (default: True)")
        parser.add_argument("--no-mask-junk", dest="mask_junk", action="store_false",
                            help="Disable junk token masking")
        parser.add_argument("--api-mode", action="store_true", default=False,
                            help="Run in headless/piped API mode without requiring a TTY")
        parser.add_argument("--max-context-tokens", type=int, default=1800,
                            help="Max tokens allocated to past conversation history (default: 1800)")
        args = parser.parse_args()
        resonance_chat(args)
    except KeyboardInterrupt:
        print("\n[Resonance Chat] Exiting...")
    finally:
        if _caff and _caff.poll() is None:
            try:
                _caff.terminate()
                _caff.wait(timeout=1.0)
            except Exception:
                pass
        sys.exit(0)


