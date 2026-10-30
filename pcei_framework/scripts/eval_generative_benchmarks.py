import sys
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.pcei import PCEINexus
from src.tokenizer import PCEITokenizer

def generate_answer(nexus, tokenizer, prompt_text, device, max_tokens=45):
    prompt = f"USER: {prompt_text}\nASSISTANT:"
    ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    curr = ids.clone()
    p_len = ids.size(1)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, *_ = nexus.core(text_input=curr, diffusion_timesteps=1, override_timestep=0)
            tok_id = torch.argmax(logits[0, -1, :]).item()
            if tok_id == 50256: break
            curr = torch.cat([curr, torch.tensor([[tok_id]], device=device)], dim=-1)
            txt = tokenizer.decode(curr[0, p_len:].tolist())
            if "\nUSER:" in txt or "<|endoftext|>" in txt: break
            
    res = tokenizer.decode(curr[0, p_len:].tolist())
    for stop in ["\nUSER:", "<|endoftext|>", "USER:"]:
        if stop in res: res = res.split(stop)[0]
    return res.strip()

def evaluate_benchmarks(checkpoint_path):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f'Error: Checkpoint {checkpoint_path} not found.')
        return
        
    print('=' * 80)
    print(f'  PCEI 750M GENERATION BENCHMARK EVALUATION')
    print(f'  Target Checkpoint: {ckpt_path.name}')
    print(f'  Device: {device.upper()} | Pure Argmax Neural Logits (0 Hacks)')
    print('=' * 80)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    step = ckpt.get('step', 'unknown')
    nexus = PCEINexus(config=cfg, device=device)
    nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
    nexus.to(device)
    nexus.eval()
    
    tokenizer = PCEITokenizer()
    bench_dir = PROJECT_ROOT / 'data' / 'benchmarks'
    
    # ── Benchmark 1: Factual Generation ──
    print(f'\n[*] Running Benchmark 1: Open Factual Generation...')
    facts_file = bench_dir / 'gen_bench_facts.jsonl'
    fact_correct = 0
    fact_total = 0
    with open(facts_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            prompt = item['prompt']
            expected = item['expected']
            ans = generate_answer(nexus, tokenizer, prompt, device)
            match = any(e.lower() in ans.lower() for e in expected)
            if match: fact_correct += 1
            fact_total += 1
            print(f'  Q: {prompt}')
            print(f'  A: {ans}  [{"PASS" if match else "FAIL"}]\n')
            
    fact_acc = (fact_correct / fact_total) * 100.0
    
    # ── Benchmark 2: Conversational Dialogue Generation ──
    print(f'\n[*] Running Benchmark 2: Human Conversational Dialogue...')
    conv_file = bench_dir / 'gen_bench_conversation.jsonl'
    conv_correct = 0
    conv_total = 0
    with open(conv_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            prompt = item['prompt']
            keywords = item['keywords']
            ans = generate_answer(nexus, tokenizer, prompt, device, max_tokens=50)
            
            # Check fluency: non-empty, contains letters, no excessive word-repetition loops
            words = ans.lower().split()
            repetition = False
            for w in set(words):
                if words.count(w) > 6 and len(w) > 3:
                    repetition = True
                    break
            has_kw = any(kw.lower() in ans.lower() for kw in keywords)
            passed = len(ans) > 5 and not repetition and has_kw
            if passed: conv_correct += 1
            conv_total += 1
            print(f'  User: {prompt}')
            print(f'  PCEI: {ans}  [{"PASS" if passed else "FAIL"}]\n')
            
    conv_acc = (conv_correct / conv_total) * 100.0
    
    # ── Benchmark 3: Mathematical Reasoning Generation ──
    print(f'\n[*] Running Benchmark 3: Arithmetic & Reasoning Generation...')
    math_file = bench_dir / 'gen_bench_math.jsonl'
    math_correct = 0
    math_total = 0
    with open(math_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            prompt = item['prompt']
            target = item['target']
            ans = generate_answer(nexus, tokenizer, prompt, device)
            
            # Strict answer extraction: Look for "is X", "= X", or the last number stated as the answer
            nums = re.findall(r'\b\d+\b', ans)
            match = False
            if nums:
                # If explicit "answer is X" or "= X" exists, check that number; otherwise check the final number in the output
                explicit_matches = re.findall(r'(?:(?:answer is|=|\bis)\s*)(\d+)', ans, re.IGNORECASE)
                if explicit_matches:
                    match = (explicit_matches[-1] == target)
                else:
                    match = (nums[-1] == target)
            if match: math_correct += 1
            math_total += 1
            print(f'  Q: {prompt}')
            print(f'  A: {ans}  (Target: {target}) [{"PASS" if match else "FAIL"}]\n')
            
    math_acc = (math_correct / math_total) * 100.0
    
    print('=' * 80)
    print(f'  BENCHMARK SUMMARY RESULTS @ Step {step}')
    print(f'  1. Open Factual QA Accuracy:       {fact_acc:5.1f}% ({fact_correct}/{fact_total})')
    print(f'  2. Conversational Coherence Score: {conv_acc:5.1f}% ({conv_correct}/{conv_total})')
    print(f'  3. Arithmetic Reasoning Accuracy:  {math_acc:5.1f}% ({math_correct}/{math_total})')
    print('=' * 80)
    
    summary = {
        'checkpoint': ckpt_path.name,
        'step': step,
        'factual_accuracy': fact_acc,
        'conversational_score': conv_acc,
        'math_accuracy': math_acc,
        'all_passed': (fact_acc >= 60.0 and conv_acc >= 60.0 and math_acc >= 60.0)
    }
    return summary

if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/pcei_750m_full_brain_best.pt'
    evaluate_benchmarks(ckpt)
