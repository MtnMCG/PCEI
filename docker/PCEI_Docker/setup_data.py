import os
import sys
import json
import subprocess
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path("/app/pcei_framework")
RAW_DIR = ROOT / "data" / "raw"
RAW_TXT = RAW_DIR / "text"
RAW_IMG = RAW_DIR / "image"
RAW_AUD = RAW_DIR / "audio"
HQ_DIR = ROOT / "data" / "high_quality"
PROCESSED_DIR = ROOT / "data" / "processed"

for d in [RAW_TXT, RAW_IMG, RAW_AUD, HQ_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def setup_text_data(target_gb=4.0):
    print(f"\n--- Setting up Text Data ({target_gb} GB target - 1 Billion Tokens) ---")
    print("Requirement: 70% High-Quality (Logic/Code/Instruct), 30% Diverse (Wiki/Books)")
    
    hq_target = target_gb * 0.7 * (1024**3)
    diverse_target = target_gb * 0.3 * (1024**3)
    
    hq_size = 0
    diverse_size = 0
    
    # 1. Diverse Data (WikiText & Books)
    print("\n[Diverse] Loading WikiText-103...")
    ds_wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True, trust_remote_code=True)
    wiki_path = RAW_TXT / "diverse_wiki.txt"
    with open(wiki_path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds_wiki, desc="Writing WikiText"):
            f.write(entry['text'] + "\n")
            diverse_size += len(entry['text'])
            if diverse_size >= diverse_target * 0.5:
                break
                
    print("\n[Diverse] Loading BookCorpus...")
    ds_books = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
    books_path = RAW_TXT / "diverse_books.txt"
    with open(books_path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds_books, desc="Writing Books"):
            f.write(entry['text'] + "\n")
            diverse_size += len(entry['text'])
            if diverse_size >= diverse_target:
                break

    # 2. High-Quality Data (Code & Instruction/Reasoning)
    print("\n[High-Quality] Loading Code Search Net (Python)...")
    ds_code = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
    code_path = HQ_DIR / "hq_python_code.jsonl"
    with open(code_path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds_code, desc="Writing Python Code"):
            f.write(json.dumps({"text": entry['whole_func_string']}) + "\n")
            hq_size += len(entry['whole_func_string'])
            if hq_size >= hq_target * 0.5:
                break
                
    print("\n[High-Quality] Loading OpenAssistant (Instruct/Reasoning)...")
    ds_instruct = load_dataset("OpenAssistant/oasst1", split="train", streaming=True, trust_remote_code=True)
    instruct_path = HQ_DIR / "hq_instruct.jsonl"
    with open(instruct_path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds_instruct, desc="Writing Instruct"):
            f.write(json.dumps({"text": entry['text']}) + "\n")
            hq_size += len(entry['text'])
            if hq_size >= hq_target:
                break
                
    print(f"\n✅ Text setup complete. HQ: {hq_size/(1024**3):.2f}GB, Diverse: {diverse_size/(1024**3):.2f}GB")

def setup_multimodal_data():
    print("\n--- Setting up Multimodal Data ---")
    print("Using acquire_massive_data.py to pull COCO/Clotho/LibriSpeech...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "acquire_massive_data.py"), "--subset-size", "10000"], cwd=ROOT, env=env)

if __name__ == "__main__":
    setup_text_data(target_gb=4.0)
    setup_multimodal_data()
    print("\n✅ All 1B tokens and multimodal data acquired and formatted.")
