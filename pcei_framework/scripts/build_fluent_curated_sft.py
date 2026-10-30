import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
data_dir = PROJECT_ROOT / "data"

def clean_tag(text: str) -> str:
    text = text.strip()
    for tag in ["USER:", "ASSISTANT:", "SYSTEM:"]:
        if text.startswith(tag):
            text = text[len(tag):].strip()
    for tag in ["\nUSER:", "\nASSISTANT:", "ASSISTANT:", "USER:"]:
        if text.endswith(tag):
            text = text[:-len(tag)].strip()
    return text.strip()

samples = []

# 1. High-Precision Arithmetic (Direct, simple math without convoluted run-on steps)
arithmetic_pairs = []
for a in range(0, 21):
    for b in range(0, 21):
        arithmetic_pairs.append((f"What is {a} + {b}?", f"{a} + {b} = {a + b}."))
        arithmetic_pairs.append((f"{a} + {b} =", f"{a + b}"))
        if a >= b:
            arithmetic_pairs.append((f"What is {a} - {b}?", f"{a} - {b} = {a - b}."))
            arithmetic_pairs.append((f"{a} - {b} =", f"{a - b}"))
for a in range(1, 13):
    for b in range(1, 13):
        arithmetic_pairs.append((f"What is {a} * {b}?", f"{a} * {b} = {a * b}."))
        arithmetic_pairs.append((f"{a} * {b} =", f"{a * b}"))
        arithmetic_pairs.append((f"What is {a * b} / {a}?", f"{a * b} / {a} = {b}."))

random.shuffle(arithmetic_pairs)
for p, r in arithmetic_pairs[:1200]:
    samples.append((p, r, "arithmetic"))

# 2. Perfect Identity & Creator Grounding
identity_templates = [
    ("Who created you?", "I was created by Emerson Ramirez (mtnmcg) as part of the PCEI project."),
    ("Who made you?", "I was created and trained by Emerson Ramirez."),
    ("Who is your creator?", "My creator is Emerson Ramirez (also known as mtnmcg)."),
    ("Who built PCEI?", "PCEI was designed and built by Emerson Ramirez."),
    ("What is your name?", "I am PCEI (Physically Constrained Emergent Intelligence)."),
    ("What does PCEI stand for?", "PCEI stands for Physically Constrained Emergent Intelligence."),
    ("What are you?", "I am PCEI, an open-source neural architecture combining BitNet 1.58b ternary weights with continuous physical locality constraints."),
    ("Hi", "Hello! How can I help you today?"),
    ("Hello!", "Hey there! What are you working on today?"),
    ("whatsup", "Not much, just ready to help you code, solve problems, or chat. What's on your mind?"),
    ("What's up?", "Everything is running smoothly! How can I help you today?"),
    ("How are you doing today?", "I'm doing great, thank you! What would you like to explore today?"),
    ("Thank you", "You're very welcome! Let me know if you need anything else."),
    ("Thanks for your help", "Anytime! Happy to help.")
]
# Repeat identity templates so they have solid representation
for _ in range(30):
    for p, r in identity_templates:
        samples.append((p, r, "identity"))

# 3. Clean Factual Q&A & Geography
facts = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("What is gravity?", "Gravity is a fundamental force of nature that pulls objects with mass toward each other."),
    ("Can you explain what gravity is in simple terms?", "Gravity is the invisible force that pulls objects toward each other, keeping our feet on the ground and planets orbiting the sun."),
    ("Why is the sky blue?", "The sky is blue because Earth's atmosphere scatters sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves (Rayleigh scattering)."),
    ("What is water made of?", "Water is composed of two hydrogen atoms bonded to one oxygen atom (H2O)."),
    ("Describe why time flies like an arrow, but fruit flies like a banana?", "This is a classic English wordplay pun. In the first half, 'flies' is a verb describing how quickly time moves. In the second half, 'fruit flies' is a noun referring to the insect, which enjoys eating bananas!")
]
for _ in range(25):
    for p, r in facts:
        samples.append((p, r, "facts"))

# 4. Clean Conversational Teacher & Dialogue (from sanitized teacher)
p_teacher = data_dir / "curriculum" / "pcei_clean_teacher_conversational.jsonl"
if p_teacher.exists():
    with open(p_teacher, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            p = clean_tag(rec.get("prompt", ""))
            r = clean_tag(rec.get("response", ""))
            if p and r and len(r) < 500 and not any(bad in r for bad in ["846", "Physically Constrained"]):
                samples.append((p, r, "dialogue"))

# 5. Clean Conversational from pure_conversational_clean
p_pure = data_dir / "curriculum" / "pcei_pure_conversational_clean.jsonl"
if p_pure.exists():
    with open(p_pure, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            p = clean_tag(rec.get("prompt", ""))
            r = clean_tag(rec.get("response", ""))
            if p and r and 20 < len(r) < 600:
                samples.append((p, r, "pure_chat"))

# 6. Narrative Stories (Complete, short stories from TinyStories)
p_tiny = data_dir / "tinystories_clean.txt"
if p_tiny.exists():
    with open(p_tiny, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
        stories = [s.strip() for s in full_text.split("\n\n") if 100 < len(s.strip()) < 500]
        random.shuffle(stories)
        for s in stories[:600]:
            samples.append(("Tell me a short story.", s, "story"))
            samples.append(("Tell me a short story about a little cat.", s.replace("dog", "cat").replace("bear", "cat"), "story"))

random.shuffle(samples)

output_file = data_dir / "curriculum" / "pcei_fluent_curated_sft.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for p, r, dom in samples:
        f.write(json.dumps({"prompt": p, "response": r, "domain": dom}, ensure_ascii=False) + "\n")

print(f"Successfully assembled {len(samples):,} curated, 100% complete fluent SFT pairs in {output_file.name}!")
