#!/usr/bin/env python3
"""
scripts/generate_kindergarten_arithmetic.py

Procedurally generates a pristine kindergarten arithmetic curriculum.
Every answer is computed by Python — zero label errors, zero modular arithmetic,
zero formatting drift, zero truncation, zero web noise.

Output: data/curriculum/kindergarten_arithmetic.jsonl
"""

import json
import random
import itertools
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "curriculum" / "kindergarten_arithmetic.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── TinyStory-style characters and nouns (restricted vocabulary matching TinyStories) ──
NAMES   = ["Lily", "Tim", "Mia", "Sam", "Ben", "Lucy", "Jake", "Zoe", "Tom", "Anna",
           "Leo", "Emma", "Max", "Ella", "Jack", "Ava", "Finn", "Nora", "Owen", "Ivy"]
OBJECTS = ["apples", "cookies", "toys", "balls", "books", "pencils", "stickers",
           "flowers", "rocks", "fish", "shells", "grapes", "carrots", "ribbons", "stars"]
GIVE_VERBS   = ["gives", "hands", "passes", "shares"]
TAKE_VERBS   = ["takes", "eats", "loses", "drops"]
HAS_VERBS    = ["has", "picks up", "finds", "collects"]

def fmt_qa(prompt: str, response: str) -> dict:
    return {
        "prompt": f"USER: {prompt}\nASSISTANT:",
        "response": f" {response}"
    }

# ── FORMAT 1: Direct equation (trains token-level operator recognition) ────────
def gen_direct_equations(n: int) -> list:
    rows = []
    # Addition: 0–20 + 0–20
    for _ in range(n // 4):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        rows.append(fmt_qa(f"{a} + {b} =", f"{a + b}"))
    # Subtraction: a >= b, 0–20
    for _ in range(n // 4):
        a = random.randint(0, 20)
        b = random.randint(0, a)
        rows.append(fmt_qa(f"{a} - {b} =", f"{a - b}"))
    # Multiplication: 1–10 × 1–10
    for _ in range(n // 4):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        rows.append(fmt_qa(f"{a} * {b} =", f"{a * b}"))
    # Division: exact only, divisor 1–10
    for _ in range(n // 4):
        b = random.randint(1, 10)
        result = random.randint(1, 10)
        a = b * result
        rows.append(fmt_qa(f"{a} / {b} =", f"{result}"))
    return rows

# ── FORMAT 2: Plain English Q&A ──────────────────────────────────────────────
ENGLISH_TEMPLATES = {
    "add":  ["What is {a} plus {b}?",
             "Add {a} and {b}.",
             "What do you get when you add {a} to {b}?",
             "{a} plus {b} equals what?"],
    "sub":  ["What is {a} minus {b}?",
             "Subtract {b} from {a}.",
             "What is {a} take away {b}?",
             "{a} minus {b} equals what?"],
    "mul":  ["What is {a} times {b}?",
             "Multiply {a} by {b}.",
             "What do you get when you multiply {a} and {b}?",
             "{a} multiplied by {b} is what?"],
    "div":  ["What is {a} divided by {b}?",
             "Divide {a} by {b}.",
             "How many times does {b} go into {a}?",
             "{a} divided by {b} equals what?"],
}
ENGLISH_ANSWERS = {
    "add": "{a} + {b} = {r}.",
    "sub": "{a} - {b} = {r}.",
    "mul": "{a} * {b} = {r}.",
    "div": "{a} / {b} = {r}.",
}

def gen_english_qa(n: int) -> list:
    rows = []
    for _ in range(n // 4):
        a = random.randint(0, 20); b = random.randint(0, 20); r = a + b
        q = random.choice(ENGLISH_TEMPLATES["add"]).format(a=a, b=b)
        rows.append(fmt_qa(q, ENGLISH_ANSWERS["add"].format(a=a, b=b, r=r)))
    for _ in range(n // 4):
        a = random.randint(0, 20); b = random.randint(0, a); r = a - b
        q = random.choice(ENGLISH_TEMPLATES["sub"]).format(a=a, b=b)
        rows.append(fmt_qa(q, ENGLISH_ANSWERS["sub"].format(a=a, b=b, r=r)))
    for _ in range(n // 4):
        a = random.randint(1, 10); b = random.randint(1, 10); r = a * b
        q = random.choice(ENGLISH_TEMPLATES["mul"]).format(a=a, b=b)
        rows.append(fmt_qa(q, ENGLISH_ANSWERS["mul"].format(a=a, b=b, r=r)))
    for _ in range(n // 4):
        b = random.randint(1, 10); r = random.randint(1, 10); a = b * r
        q = random.choice(ENGLISH_TEMPLATES["div"]).format(a=a, b=b)
        rows.append(fmt_qa(q, ENGLISH_ANSWERS["div"].format(a=a, b=b, r=r)))
    return rows

# ── FORMAT 3: TinyStory-style word problems ───────────────────────────────────
def gen_word_problems(n: int) -> list:
    rows = []
    for _ in range(n // 2):
        # Addition word problem
        name = random.choice(NAMES)
        obj  = random.choice(OBJECTS)
        a    = random.randint(1, 15)
        b    = random.randint(1, 10)
        r    = a + b
        giver = random.choice([n for n in NAMES if n != name])
        give  = random.choice(GIVE_VERBS)
        has   = random.choice(HAS_VERBS)
        q = f"{name} {has} {a} {obj}. {giver} {give} {name} {b} more {obj}. How many {obj} does {name} have now?"
        ans = f"{name} has {a} + {b} = {r} {obj}."
        rows.append(fmt_qa(q, ans))
    for _ in range(n // 2):
        # Subtraction word problem
        name = random.choice(NAMES)
        obj  = random.choice(OBJECTS)
        a    = random.randint(2, 15)
        b    = random.randint(1, a)
        r    = a - b
        has  = random.choice(HAS_VERBS)
        take = random.choice(TAKE_VERBS)
        q = f"{name} {has} {a} {obj}. {name} {take} {b} {obj}. How many {obj} does {name} have left?"
        ans = f"{name} has {a} - {b} = {r} {obj} left."
        rows.append(fmt_qa(q, ans))
    return rows

# ── FORMAT 4: 2-step chains of thought ───────────────────────────────────────
def gen_two_step(n: int) -> list:
    rows = []
    for _ in range(n // 2):
        # (a + b) - c
        a = random.randint(1, 15); b = random.randint(1, 10)
        mid = a + b
        c = random.randint(1, mid)
        r = mid - c
        q = f"What is {a} + {b} - {c}?"
        ans = f"First, {a} + {b} = {mid}. Then, {mid} - {c} = {r}. The answer is {r}."
        rows.append(fmt_qa(q, ans))
    for _ in range(n // 2):
        # (a * b) + c
        a = random.randint(1, 8); b = random.randint(1, 8)
        mid = a * b
        c = random.randint(1, 10)
        r = mid + c
        q = f"What is {a} * {b} + {c}?"
        ans = f"First, {a} * {b} = {mid}. Then, {mid} + {c} = {r}. The answer is {r}."
        rows.append(fmt_qa(q, ans))
    return rows

# ── Main generation ──────────────────────────────────────────────────────────
TARGET = 50_000

n_direct   = TARGET * 30 // 100   # 30% — 15,000
n_english  = TARGET * 30 // 100   # 30% — 15,000
n_wordprob = TARGET * 25 // 100   # 25% — 12,500
n_twostep  = TARGET * 15 // 100   # 15% —  7,500

print(f"Generating {TARGET:,} kindergarten arithmetic examples...")
all_rows = []
all_rows += gen_direct_equations(n_direct)
all_rows += gen_english_qa(n_english)
all_rows += gen_word_problems(n_wordprob)
all_rows += gen_two_step(n_twostep)

# Shuffle so the streamer sees mixed formats throughout training
random.shuffle(all_rows)

# Write — one JSON object per line, no truncation, UTF-8 clean
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in all_rows:
        line = json.dumps(row, ensure_ascii=True)
        f.write(line + "\n")

print(f"Written {len(all_rows):,} examples to {OUTPUT_PATH}")
print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

# ── Sanity check: verify first 20 entries ────────────────────────────────────
print("\nSanity check (first 5 entries):")
with open(OUTPUT_PATH, "r") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        obj = json.loads(line)
        print(f"  [{i}] {obj['prompt']!r}  =>  {obj['response']!r}")

print("\nDone. Zero errors guaranteed — every answer computed by Python math engine.")
