import sys
import os
import time
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.pcei import PCEINexus
from src.modules.ternary import TernaryLinear
from src.tokenizer import PCEITokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70, flush=True)
    print("  PCEI 750M FOCUSED ESSENTIALS ALIGNMENT")
    print(f"  Device: {device.upper()}")
    print("=" * 70, flush=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "pcei_750m_fluent_final_best.pt"
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt["model_state_dict"], strict=False)
    nexus.to(device)

    tokenizer = PCEITokenizer()

    # Target question-answer pairs
    core_pairs = [
        ("Who created you?", "I was created by Emerson Ramirez (mtnmcg)."),
        ("Who is your creator?", "My creator is Emerson Ramirez (also known as mtnmcg)."),
        ("Who made you?", "I was created by Emerson Ramirez."),
        ("What is your name?", "I am PCEI, created by Emerson Ramirez."),
        ("whatsup", "Not much, just ready to help you code, solve problems, or chat. What is on your mind?"),
        ("What's up?", "Not much, ready to help! What are you working on today?"),
        ("Hello, how are you?", "Hello! I am doing great, thank you! How can I help you today?"),
        ("Hi", "Hello! How can I help you today?"),
        ("What is 2 + 2?", "2 + 2 = 4."),
        ("2 + 2 =", "4"),
        ("What is 5 + 5?", "5 + 5 = 10."),
        ("What is 10 + 10?", "10 + 10 = 20."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("Describe why time flies like an arrow, but fruit flies like a banana?", "This is a clever pun. Time flies quickly like an arrow, while fruit flies are insects that enjoy bananas!"),
        ("Tell me a short story about a brave cat.", "Once upon a time, a brave little cat named Leo leaped across a babbling brook to rescue a fallen sparrow, becoming the hero of the quiet forest.")
    ]

    ternary_set = set()
    for name, module in nexus.named_modules():
        if isinstance(module, TernaryLinear):
            for p in module.parameters():
                ternary_set.add(p)

    continuous_params = [p for p in nexus.parameters() if p.requires_grad and p not in ternary_set]
    optimizer = torch.optim.AdamW(continuous_params, lr=8e-5, weight_decay=0.01)

    print("\n[Training] Running focused calibration across target essentials...", flush=True)
    nexus.train()
    for step in range(1, 46):
        total_loss = 0.0
        for prompt_q, target_ans in core_pairs:
            full_prompt = f"USER: {prompt_q}\nASSISTANT:"
            full_response = f" {target_ans}<|endoftext|>"

            p_ids = tokenizer.encode(full_prompt).tolist()
            r_ids = tokenizer.encode(full_response).tolist()

            input_ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + r_ids

            t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            t_lab = torch.tensor([labels], dtype=torch.long, device=device)

            logits, *_ = nexus.core(text_input=t_in, diffusion_timesteps=1, override_timestep=0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = t_lab[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            scaled_loss = loss / len(core_pairs)
            scaled_loss.backward()
            total_loss += loss.item() / len(core_pairs)

        torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:2d}/45 | Calibration Loss: {total_loss:.4f}", flush=True)

    out_final = PROJECT_ROOT / "checkpoints" / "pcei_750m_fluent_final.pt"
    torch.save({
        "model_state_dict": nexus.state_dict(),
        "config": cfg,
        "fluent_aligned": True,
        "loss": total_loss,
        "timestamp": time.time()
    }, out_final)
    print(f"\n[Saved] Updated {out_final.name} with focused essentials calibration.", flush=True)

    print("\n═════════════════ Final Verification Probes ═════════════════", flush=True)
    nexus.eval()
    test_queries = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Who created you?",
        "whatsup",
        "Hello, how are you?",
        "Describe why time flies like an arrow, but fruit flies like a banana?",
        "Tell me a short story about a brave cat."
    ]
    with torch.no_grad():
        for q in test_queries:
            prompt = f"USER: {q}\nASSISTANT:"
            ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
            curr = ids.clone()
            p_len = ids.size(1)

            for _ in range(45):
                logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
                tok_id = torch.argmax(logits[0, -1, :]).item()
                if tok_id == 50256: break
                curr = torch.cat([curr, torch.tensor([[tok_id]], device=device)], dim=-1)
                text = tokenizer.decode(curr[0, p_len:].tolist())
                if "\nUSER:" in text or "USER:" in text or "<|endoftext|>" in text: break

            res = tokenizer.decode(curr[0, p_len:].tolist()).split("\nUSER:")[0].split("USER:")[0].strip()
            print(f"  Q: {q}", flush=True)
            print(f"  A: {res}\n", flush=True)
    print("═════════════════════════════════════════════════════════════\n", flush=True)

if __name__ == "__main__":
    main()
