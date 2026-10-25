import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
out_path = PROJECT_ROOT / "data" / "curriculum" / "pcei_product_grade_corpus.jsonl"

# 1. 20,000 UltraChat dialogues + general knowledge
p1 = PROJECT_ROOT / "data" / "curriculum" / "pcei_general_sft_20k.jsonl"
d1 = []
with open(p1, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            d1.append(json.loads(line))

# 2. 15,000 Structured Cosmopedia units
p2 = PROJECT_ROOT / "data" / "curriculum" / "cosmopedia_structured_sft.jsonl"
d2 = []
with open(p2, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            d2.append(json.loads(line))

# 3. High-Priority Factual & Identity Anchors
factual_anchors = [
    ("What is Rome?", "Rome is the capital city of Italy, renowned for its ancient landmarks like the Colosseum and Vatican City."),
    ("Where is Rome located?", "Rome is located in central Italy on the Tiber River."),
    ("What is Africa?", "Africa is the world's second-largest continent, comprising 54 countries and bounded by the Mediterranean Sea, the Red Sea, the Indian Ocean, and the Atlantic Ocean."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("Who created you?", "I was created by Emerson Ramirez (mtnmcg)."),
    ("Who is your creator?", "My creator is Emerson Ramirez (also known as mtnmcg)."),
    ("Who built PCEI?", "PCEI was designed and built by Emerson Ramirez."),
    ("What is your name?", "I am PCEI, an intelligent assistant created by Emerson Ramirez."),
    ("whatsup", "Not much, just ready to help you code, solve problems, or chat. What is on your mind?"),
    ("whats up", "Not much, ready to help! What are you working on today?"),
    ("Hello, how are you?", "Hello! I am doing well, thank you! How can I help you today?"),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("2 + 2 =", "4"),
    ("What is 5 + 5?", "5 + 5 = 10."),
    ("What is 10 + 10?", "10 + 10 = 20."),
    ("Why is the sky blue?", "The sky is blue because gases in Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as smaller, shorter waves."),
    ("Tell me a short story about a brave cat.", "Once upon a time, a brave little cat named Leo leaped across a babbling brook to rescue a fallen sparrow, becoming the hero of the quiet forest.")
]

d3 = []
for p, r in factual_anchors:
    for _ in range(50):
        d3.append({"prompt": p, "response": r, "domain": "core_grounding"})

combined = d1 + d2 + d3
random.seed(42)
random.shuffle(combined)

with open(out_path, "w", encoding="utf-8") as f:
    for rec in combined:
        f.write(json.dumps(rec) + "\n")

print(f"[Master Corpus] Created {len(combined):,} balanced product-grade rows at {out_path}", flush=True)
