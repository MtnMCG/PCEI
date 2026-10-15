"""
PCEI Synthetic Data Generator — Structured Patterns for Grokking

Generates datasets known to produce grokking behavior:
1. Modular arithmetic: a OP b = c (mod p) — the classic grokking dataset
2. Sequence completion: ABC→DEF patterns
3. Ollama-generated reasoning data (optional)

Output: JSONL files in data/synthetic/
"""
import json
import random
import argparse
import itertools
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════
# 1. MODULAR ARITHMETIC (Classic Grokking)
# ═══════════════════════════════════════
def generate_modular_arithmetic(
    p: int = 97,  # prime modulus
    operations: List[str] = ["+", "-", "*"],
    output_path: Path = None,
):
    """
    Generates all (a OP b) mod p examples.
    This is THE canonical grokking dataset from Neel Nanda's work.
    The model memorizes first, then suddenly generalizes.
    """
    if output_path is None:
        output_path = DATA_DIR / "modular_arithmetic.jsonl"
    
    records = []
    for op in operations:
        for a in range(p):
            for b in range(p):
                if op == "+":
                    result = (a + b) % p
                elif op == "-":
                    result = (a - b) % p
                elif op == "*":
                    result = (a * b) % p
                else:
                    continue
                
                # Format as text the model can learn
                text = f"{a} {op} {b} = {result}"
                records.append({
                    "text": text,
                    "type": "modular_arithmetic",
                    "a": a, "b": b, "op": op, "result": result, "p": p
                })
    
    # Shuffle
    random.shuffle(records)
    
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    print(f"[ModularArithmetic] Generated {len(records)} examples (p={p}, ops={operations})")
    print(f"  → {output_path}")
    return records


# ═══════════════════════════════════════
# 2. SEQUENCE PATTERNS (Learnable structure)
# ═══════════════════════════════════════
def generate_sequence_patterns(
    num_examples: int = 5000,
    output_path: Path = None,
):
    """
    Generates various learnable patterns:
    - Repetition: ABCABC → next is A
    - Counting: 1 2 3 4 → 5 6 7 8
    - Reversal: ABCD → DCBA
    - Simple cipher: shift each char by N
    """
    if output_path is None:
        output_path = DATA_DIR / "sequence_patterns.jsonl"
    
    records = []
    
    for _ in range(num_examples):
        pattern_type = random.choice(["repeat", "count", "reverse", "cipher"])
        
        if pattern_type == "repeat":
            # ABCABC → ABC
            length = random.randint(2, 8)
            base = ''.join(random.choices("abcdefghij", k=length))
            repeats = random.randint(2, 4)
            text = f"repeat: {''.join([base]*repeats)} next: {base}"
            
        elif pattern_type == "count":
            # 3 6 9 12 → 15 18
            start = random.randint(1, 20)
            step = random.randint(1, 5)
            seq_len = random.randint(4, 8)
            nums = [start + step * i for i in range(seq_len)]
            next_nums = [start + step * i for i in range(seq_len, seq_len + 3)]
            text = f"count: {' '.join(map(str, nums))} next: {' '.join(map(str, next_nums))}"
            
        elif pattern_type == "reverse":
            length = random.randint(3, 10)
            word = ''.join(random.choices("abcdefghijklmnop", k=length))
            text = f"reverse: {word} answer: {word[::-1]}"
            
        elif pattern_type == "cipher":
            shift = random.randint(1, 5)
            length = random.randint(3, 8)
            plain = ''.join(random.choices("abcdefghijklmnop", k=length))
            cipher = ''.join(chr((ord(c) - ord('a') + shift) % 26 + ord('a')) for c in plain)
            text = f"shift{shift}: {plain} answer: {cipher}"
        
        records.append({"text": text, "type": pattern_type})
    
    random.shuffle(records)
    
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    print(f"[SequencePatterns] Generated {len(records)} examples")
    print(f"  → {output_path}")
    return records


# ═══════════════════════════════════════
# 3. OLLAMA REASONING DATA
# ═══════════════════════════════════════
def generate_ollama_reasoning(
    num_examples: int = 100,
    model: str = "gpt-oss:120b-cloud",
    output_path: Path = None,
):
    """
    Uses the 120B Beast model to generate high-fidelity training data.
    Ensures every response ends with <|endoftext|> to teach termination.
    """
    if output_path is None:
        output_path = DATA_DIR / "beast_reasoning.jsonl"
    
    import requests
    
    # High-quality categories for PCEI performance testing
    topics = [
        "complex modular arithmetic word problems",
        "quantum physics conceptual explanations for AI",
        "advanced python coding challenges with docstrings",
        "multimodal synthesis: describing an image that represents a sound",
        "self-reflective reasoning on metabolic neural efficiency",
        "mathematical proofs involving prime numbers",
        "philosophical dialogues between a human and a PCEI entity",
        "general conversational flow and social intelligence",
        "common sense world knowledge and everyday reasoning"
    ]
    
    records = []
    
    for i in range(num_examples):
        topic = random.choice(topics)
        # We explicitly ask the model to produce a structured output
        # then we append the physical EOS token for the tokenizer to pick up.
        prompt = (
            f"Generate a highly detailed {topic}. "
            f"Include a 'Prompt:' and a 'Response:' section. "
            f"The response must be thorough and technically accurate. "
            f"Do not include any other text."
        )
        
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {"num_predict": 512, "temperature": 0.7}
                },
                timeout=120 # 120B model can be slow
            )
            if resp.status_code == 200:
                raw_text = resp.json().get("response", "").strip()
                if raw_text:
                    # Append EOS token to teach the model how to end a thought
                    final_text = f"{raw_text}\n<|endoftext|>"
                    records.append({
                        "text": final_text, 
                        "type": "high_fidelity_reasoning", 
                        "topic": topic,
                        "model": model
                    })
                    if (i + 1) % 5 == 0:
                        print(f"  Generated {i+1}/{num_examples}...")
                        # Intermediate save
                        with open(output_path, 'a') as f:
                            for r in records[-5:]:
                                f.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"  [Warning] Ollama call failed at step {i}: {e}")
            continue
    
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    print(f"[OllamaReasoning] Generated {len(records)} examples")
    print(f"  → {output_path}")
    return records


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCEI Synthetic Data Generator")
    parser.add_argument("--modular", action="store_true", help="Generate modular arithmetic data")
    parser.add_argument("--patterns", action="store_true", help="Generate sequence pattern data")
    parser.add_argument("--ollama", action="store_true", help="Generate Ollama reasoning data")
    parser.add_argument("--all", action="store_true", help="Generate all types")
    parser.add_argument("--prime", type=int, default=23, help="Prime modulus for modular arithmetic (smaller = faster grokking)")
    parser.add_argument("--num-patterns", type=int, default=5000, help="Number of sequence patterns")
    parser.add_argument("--num-ollama", type=int, default=50, help="Number of Ollama examples")
    parser.add_argument("--model", type=str, default="gpt-oss:120b-cloud", help="Ollama model")
    
    args = parser.parse_args()
    
    if not any([args.modular, args.patterns, args.ollama, args.all]):
        args.all = True
    
    if args.modular or args.all:
        generate_modular_arithmetic(p=args.prime)
    
    if args.patterns or args.all:
        generate_sequence_patterns(num_examples=args.num_patterns)
    
    if args.ollama or args.all:
        generate_ollama_reasoning(num_examples=args.num_ollama, model=args.model)
    
    print("\n✅ Data generation complete!")
    print(f"   Files in: {DATA_DIR}")
