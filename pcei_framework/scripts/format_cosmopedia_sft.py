import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
cosmo_path = PROJECT_ROOT / "data" / "raw" / "text" / "mass_reasoning" / "cosmopedia_textbooks.txt"
out_path = PROJECT_ROOT / "data" / "curriculum" / "cosmopedia_structured_sft.jsonl"

print("[Processor] Reading and segmenting Cosmopedia into clean paragraph units...", flush=True)

records = []
with open(cosmo_path, "r", encoding="utf-8", errors="ignore") as f:
    buffer = ""
    for line in f:
        line_s = line.strip()
        if not line_s:
            if buffer.strip():
                # Process completed paragraph
                p_text = buffer.strip()
                buffer = ""
                # Filter for high-quality informative paragraphs
                if 120 <= len(p_text) <= 550 and not p_text.startswith("==") and not p_text.startswith("#"):
                    sentences = [s.strip() for s in p_text.split(".") if s.strip()]
                    if len(sentences) >= 2:
                        first_s = sentences[0]
                        rest_s = ". ".join(sentences[1:]) + "."
                        # Create natural topical continuation / instructional prompt
                        prompt = f"Explain the concept: {first_s}."
                        response = rest_s
                        records.append({
                            "prompt": prompt,
                            "response": response,
                            "domain": "cosmopedia_knowledge"
                        })
                        if len(records) >= 15000:
                            break
            continue
        buffer += " " + line_s

print(f"[Processor] Extracted {len(records):,} structured educational units from Cosmopedia.", flush=True)

random.seed(42)
random.shuffle(records)

with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"[Processor] Saved to {out_path}", flush=True)
