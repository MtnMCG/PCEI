import json
from pathlib import Path
syn_dir = Path("data/synthetic")
raw_dir = Path("data/raw/text")
texts = []
if syn_dir.exists():
    for f in syn_dir.glob("*.jsonl"):
        with open(f) as fp:
            for line in fp:
                if line.strip():
                    try:
                        d=json.loads(line)
                        if "text" in d and d["text"]: texts.append(d["text"])
                    except: pass
if raw_dir.exists():
    for f in raw_dir.glob("*.txt"):
        content = f.read_text().strip()
        if content:
            chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 10]
            texts.extend(chunks)
print(f"Total Text Examples in Phase 1: {len(texts)}")
