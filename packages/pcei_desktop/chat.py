#!/usr/bin/env python3
"""
PCEI Desktop Chat — Standalone Interactive Dialogue Interface
Platform: Cross-Platform (macOS MPS, Linux CUDA/CPU, Windows CUDA/CPU)
Model: PCEI 750M BitNet 1.58b Ternary Mamba Architecture
"""

import os
import sys
import argparse
from pathlib import Path

# Add package directory to path
PACKAGE_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

HF_REPO = "MtnMCG/PCEI-Text"
HF_FILENAME = "pcei_750m_competitive_best.pt"
LOCAL_CKPT = PACKAGE_ROOT / "checkpoints" / "pcei_750m_competitive_best.pt"


def ensure_checkpoint(ckpt_path: Path):
    """Ensure the model weights exist locally. If not, auto-download from Hugging Face."""
    if ckpt_path.exists():
        return ckpt_path
    
    print("=" * 64)
    print("  PCEI 750M Weights Not Detected Locally")
    print(f"  Auto-downloading '{HF_FILENAME}' from Hugging Face ({HF_REPO})...")
    print("=" * 64, flush=True)
    
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=str(ckpt_path.parent),
            local_dir_use_symlinks=False
        )
        print(f"[+] Download complete: {downloaded}\n", flush=True)
        return Path(downloaded)
    except Exception as e:
        print(f"\n[!] Error auto-downloading from Hugging Face: {e}")
        print(f"    Please manually place '{HF_FILENAME}' into:\n    {ckpt_path.parent}\n")
        sys.exit(1)


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


def main():
    parser = argparse.ArgumentParser(description="PCEI 750M Standalone Terminal Dialogue Interface")
    parser.add_argument("--ckpt", type=str, default=str(LOCAL_CKPT), help="Path to checkpoint file")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (default: 0.5)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty (default: 1.05)")
    parser.add_argument("--max-new-tokens", type=int, default=248, help="Maximum generated tokens per response")
    parser.add_argument("--single-turn", action="store_true", help="Disable multi-turn conversation memory")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    args = parser.parse_args()

    # Determine fastest available GPU accelerator (CUDA -> MPS -> CPU)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        hw_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    elif device == "mps":
        hw_label = "MPS GPU (Apple Silicon Unified Metal)"
    else:
        hw_label = "CPU (No CUDA/MPS GPU detected)"

    # Verify/download weights
    ckpt_path = ensure_checkpoint(Path(args.ckpt))

    print("=" * 64)
    print("  PCEI 750M Resonance Chat — Standalone Desktop Edition")
    print("  Architecture: 24-Layer Mamba SSM + FastKAN + BitNet 1.58b Ternary")
    print(f"  Accelerator: {hw_label}")
    print(f"  Temperature: {args.temperature} | Top-p: {args.top_p} | Multi-Turn Memory: {'OFF' if args.single_turn else 'ON'}")
    print("=" * 64)

    tokenizer = PCEITokenizer()
    syntax_ids = get_syntax_token_ids(tokenizer)
    bad_tensor = get_parasitic_junk_tokens(tokenizer, device=device)

    config = PCEIConfig.load()
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio

    print(f"[*] Initializing model topology ({layers_text} Text Layers, D={d_text})...", flush=True)
    nexus = PCEINexus(config=config, device=device)
    
    print(f"[*] Loading model checkpoint: {ckpt_path.name}...", flush=True)
    ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    nexus.load_state_dict(ckpt_dict['model_state_dict'], strict=False)
    nexus.load_ambient_state(ckpt_dict)
    del ckpt_dict

    nexus.to(device)
    nexus.eval()
    print("[+] Model ready. Type your prompt below. Commands: /clear, /history, /exit\n")

    conversation_history = []
    MAX_PROMPT_TOKENS = 1800

    while True:
        try:
            user_input = input("YOU: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", ":q", "/exit", "/quit", "bye", "/bye"]:
                print("\n[PCEI] Session closed. Farewell!")
                break
            elif user_input.lower() in ["/clear", "/reset"]:
                conversation_history.clear()
                print("[Memory] Conversation history cleared.\n")
                continue
            elif user_input.lower() == "/history":
                print(f"\n[History] {len(conversation_history)} turn(s) in active context:")
                for idx, (u, a) in enumerate(conversation_history, 1):
                    print(f"  [{idx}] YOU: {u}")
                    print(f"      MODEL: {a}")
                print()
                continue
            elif user_input.lower() in ["/help", "help"]:
                print("\nCommands:")
                print("  /clear   - Clear multi-turn conversation memory")
                print("  /history - Show active conversation turns")
                print("  /exit    - Exit chat\n")
                continue

            # Build multi-turn context
            turns_to_use = [] if args.single_turn else list(conversation_history)
            while True:
                context_blocks = []
                for u_past, a_past in turns_to_use:
                    context_blocks.append(f"USER: {u_past}\nASSISTANT: {a_past}<|endoftext|>")
                context_blocks.append(f"USER: {user_input}\nASSISTANT:")
                formatted_prompt = "\n".join(context_blocks)
                prompt_ids = tokenizer.enc.encode(formatted_prompt, allowed_special={"<|endoftext|>"})
                if len(prompt_ids) <= MAX_PROMPT_TOKENS or not turns_to_use:
                    break
                turns_to_use.pop(0)

            print("MODEL: ", end="", flush=True)
            generated_ids = []

            with torch.inference_mode():
                curr_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                nexus.response_gate.reset()

                for step in range(args.max_new_tokens):
                    logits, *_ = nexus.core(
                        text_input=curr_tokens,
                        diffusion_timesteps=1,
                        override_timestep=0,
                        slow_state=getattr(nexus, 'ambient_field', None) and nexus.ambient_field.slow_state,
                    )
                    logits_last = logits[:, -1, :].squeeze(0).clone().float()

                    # Mask out parasitic junk tokens
                    logits_last[bad_tensor] = -1e4

                    # Consecutive non-syntax repeat ban
                    if len(generated_ids) > 0:
                        last_tok = generated_ids[-1]
                        if last_tok not in syntax_ids:
                            logits_last[last_tok] = -1e4

                    # Multiplicative repetition penalty
                    if args.repetition_penalty > 1.0 and len(generated_ids) > 0:
                        recent_unique = set(generated_ids[-40:])
                        for t in recent_unique:
                            if t not in syntax_ids:
                                if logits_last[t] > 0:
                                    logits_last[t] /= args.repetition_penalty
                                else:
                                    logits_last[t] *= args.repetition_penalty

                    # Temperature & Top-p nucleus filtering
                    scaled = logits_last / max(1e-4, args.temperature)
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
                        break

                    generated_ids.append(next_token_id)
                    word = tokenizer.enc.decode([next_token_id])

                    full_gen = tokenizer.enc.decode(generated_ids)
                    if any(stop in full_gen for stop in ["\nUSER:", "\nHuman:", "<|endoftext|>", "USER:"]):
                        break

                    print(word, end="", flush=True)
                    curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)

            print()
            full_gen_clean = tokenizer.enc.decode(generated_ids)
            for stop in ["\nUSER:", "\nHuman:", "<|endoftext|>", "USER:"]:
                if stop in full_gen_clean:
                    full_gen_clean = full_gen_clean.split(stop)[0]
            clean_resp = full_gen_clean.strip()
            if clean_resp:
                conversation_history.append((user_input, clean_resp))

        except (KeyboardInterrupt, EOFError):
            print("\n\n[PCEI] Session closed. Farewell!")
            break


if __name__ == "__main__":
    main()
