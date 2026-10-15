"""
download_claude_dataset.py
Downloads Havoc999/The-Claude-Dataset as a local JSONL file,
applying phased CoT splitting (reasoning + response as separate records).
Run once before training. Safe to resume if interrupted.
"""
import json
import re
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "data" / "synthetic" / "anthropic_claude.jsonl"
TARGET_CONVERSATIONS = 10000  # Each conversation produces up to 2 records

def parse_cot(assistant_content: str):
    match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        after = assistant_content[match.end():].strip()
        return reasoning, after
    return None, assistant_content

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Run: pip install datasets")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Count existing records to support resume
    existing = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'r') as f:
            existing = sum(1 for _ in f)
        print(f"Resuming — {existing} records already on disk.")

    print(f"Streaming from Havoc999/The-Claude-Dataset...")
    ds = load_dataset('Havoc999/The-Claude-Dataset', split='train', streaming=True)

    count = 0
    written = 0
    with open(OUTPUT_PATH, 'a') as out:
        for row in ds:
            if count >= TARGET_CONVERSATIONS:
                break

            msgs = row.get('messages', [])
            prompt = ''
            assistant_content = ''
            for m in msgs:
                if m['role'] == 'system':
                    prompt += f"SYSTEM: {m['content']}\n"
                elif m['role'] == 'user':
                    prompt += f"{m['content']}\n"
                elif m['role'] == 'assistant':
                    assistant_content = m['content']
                    break

            if not prompt or not assistant_content:
                count += 1
                continue

            prompt = prompt.strip()
            reasoning, response = parse_cot(assistant_content)

            records = []
            if reasoning and response:
                records.append({
                    "prompt": prompt + "\nASSISTANT:",
                    "response": f" <think>\n{reasoning}\n</think>\n<|endoftext|>",
                    "category": "anthropic_reasoning"
                })
                reasoning_ctx = reasoning[:800]
                records.append({
                    "prompt": f"<think>\n{reasoning_ctx}\n</think>\nASSISTANT:",
                    "response": f" {response.strip()}\n<|endoftext|>",
                    "category": "anthropic_response"
                })
            else:
                records.append({
                    "prompt": prompt + "\nASSISTANT:",
                    "response": f" {assistant_content.strip()}\n<|endoftext|>",
                    "category": "anthropic_direct"
                })

            # Skip records already written (resume support)
            for rec in records:
                if written < existing:
                    written += 1
                    continue
                out.write(json.dumps(rec) + "\n")
                written += 1

            count += 1
            if count % 100 == 0:
                print(f"  {count}/{TARGET_CONVERSATIONS} conversations → {written} records", flush=True)

    print(f"\nDone. {written} total records at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
