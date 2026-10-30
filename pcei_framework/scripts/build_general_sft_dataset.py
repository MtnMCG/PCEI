import json
import random
from pathlib import Path
from datasets import load_dataset

out_path = Path("data/curriculum/pcei_general_sft_20k.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

print("[Builder] Extracting multi-turn conversational turns from UltraChat 200k...", flush=True)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:35000]")

records = []
for item in ds:
    msgs = item.get("messages", [])
    for i in range(len(msgs) - 1):
        if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
            u_text = msgs[i].get("content", "").strip()
            a_text = msgs[i+1].get("content", "").strip()
            # Filter clean conversational lengths suitable for 750M sequence window
            if 10 <= len(u_text) <= 500 and 15 <= len(a_text) <= 650:
                if "```" not in a_text and "|---" not in a_text and "http" not in a_text:
                    records.append({
                        "prompt": u_text,
                        "response": a_text,
                        "domain": "general_chat"
                    })
    if len(records) >= 20000:
        break

print(f"[Builder] Extracted {len(records):,} clean conversation pairs from UltraChat.", flush=True)

print("[Builder] Extracting 2,500 TinyStories narrative pairs...", flush=True)
ts = load_dataset("roneneldan/TinyStories", split="train[:5000]")
story_count = 0
for item in ts:
    text = item["text"].strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paras:
        p_text = paras[0]
        if 80 <= len(p_text) <= 450:
            sentences = [s.strip() for s in p_text.split(".") if s.strip()]
            if len(sentences) >= 2:
                records.append({
                    "prompt": f"Write a short story about {sentences[0].lower()}.",
                    "response": ". ".join(sentences[1:]) + ".",
                    "domain": "narrative"
                })
                story_count += 1
                if story_count >= 2500:
                    break

print(f"[Builder] Extracted {story_count:,} story pairs.", flush=True)

print("[Builder] Adding factual knowledge, science, arithmetic, and creator identity...", flush=True)
factual_seeds = [
    # Geography & General Knowledge
    ("What is Rome?", "Rome is the capital city of Italy, world-renowned for its rich history, ancient Roman architecture, and the Vatican."),
    ("Where is Rome located?", "Rome is located in the central-western portion of the Italian Peninsula, within the Lazio region of Italy."),
    ("What is Africa?", "Africa is the world's second-largest and second-most-populous continent, home to 54 recognized countries and immense cultural diversity."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the Pacific Ocean?", "The Pacific Ocean is the largest and deepest of the world's oceanic divisions, extending from the Arctic to the Southern Ocean."),
    ("What is the Sahara?", "The Sahara is the largest hot desert in the world, spanning most of North Africa."),
    
    # Science & Definitions
    ("What is gravity?", "Gravity is a fundamental natural force that pulls objects with mass or energy toward one another."),
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter shorter wavelengths of sunlight (blue and violet) more than longer wavelengths."),
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants and organisms use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."),
    ("What is water made of?", "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom (H2O)."),
    ("What is an atom?", "An atom is the basic building block of chemistry, composed of a nucleus of protons and neutrons surrounded by electrons."),
    
    # Creator & Identity
    ("Who created you?", "I was created by Emerson Ramirez (mtnmcg)."),
    ("Who is your creator?", "My creator is Emerson Ramirez (also known as mtnmcg)."),
    ("Who built PCEI?", "PCEI was designed and built by Emerson Ramirez."),
    ("What is your name?", "I am PCEI, an emergent intelligence assistant created by Emerson Ramirez."),
    
    # Conversational Greetings
    ("whatsup", "Not much, just ready to help you code, solve problems, or chat. What is on your mind?"),
    ("whats up", "Not much, ready to help! What are you working on today?"),
    ("What's up?", "Not much! How can I help you today?"),
    ("Hello, how are you?", "Hello! I am doing well, thank you! How can I assist you today?"),
    ("Hi", "Hello! How can I help you today?"),
    
    # Elementary Arithmetic
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("2 + 2 =", "4"),
    ("What is 5 + 5?", "5 + 5 = 10."),
    ("What is 10 + 10?", "10 + 10 = 20."),
    ("What is 7 * 8?", "7 * 8 = 56."),
    ("What is 100 / 10?", "100 / 10 = 10.")
]

for p, r in factual_seeds:
    for _ in range(35):
        records.append({"prompt": p, "response": r, "domain": "factual_grounding"})

random.seed(42)
random.shuffle(records)

with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"[Builder] Success! Generated {len(records):,} balanced pairs in {out_path}", flush=True)
