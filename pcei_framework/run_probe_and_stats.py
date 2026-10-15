#!/usr/bin/env python3
"""
Quick probe + data stats runner.
Run: venv/bin/python run_probe_and_stats.py
"""
import json, pathlib, sys, time, os
sys.path.insert(0, str(pathlib.Path(__file__).parent))

ROOT = pathlib.Path(__file__).parent

# ── 1. DATA STATS ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DATA INVENTORY")
print("="*60)

JSONL_DIRS = [
    ROOT / "data" / "synthetic",
    ROOT / "data" / "curriculum",
    ROOT / "data" / "high_quality",
]

total_records = 0
total_bytes   = 0

for d in JSONL_DIRS:
    if not d.exists():
        continue
    for f in sorted(d.glob("*.jsonl")):
        count = 0
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # count as usable if it has any of these fields
                    if any(k in obj for k in ("text","prompt","response","messages","content","output","instruction")):
                        count += 1
                except Exception:
                    pass
        size_kb = f.stat().st_size / 1024
        total_records += count
        total_bytes   += f.stat().st_size
        print(f"  {f.parent.name}/{f.name:<40s}  {count:>7,} records  ({size_kb:,.0f} KB)")

total_mb = total_bytes / (1024**2)
# Rough estimate: GPT-2 tokenizer averages ~4.1 chars/token for English prose
avg_chars_per_token = 4.1
# Average chars per record (rough): total_bytes / total_records
if total_records > 0:
    avg_chars = total_bytes / total_records
    est_tokens = int(total_bytes / avg_chars_per_token)
else:
    est_tokens = 0

print(f"\n  TOTAL: {total_records:,} records | {total_mb:.1f} MB raw | ~{est_tokens/1e6:.1f}M estimated tokens")

# Chinchilla scaling law for 750M params: ~15B tokens for "fully trained"
needed_tokens = 15_000_000_000
pct = (est_tokens / needed_tokens) * 100
print(f"  Chinchilla budget for 750M params: ~15B tokens")
print(f"  Coverage: {pct:.3f}% of required data")
print()

# ── 2. PROBE THE SUPERCHARGED CHECKPOINT ──────────────────────────────────────
print("="*60)
print("  RUNNING PROBE ON pcei_750m_supercharged.pt")
print("="*60)

import subprocess, sys as _sys
result = subprocess.run(
    [str(ROOT / "venv" / "bin" / "python"),
     str(ROOT / "scripts" / "probe_post_graft.py"),
     "pcei_750m_supercharged.pt"],
    cwd=str(ROOT),
    capture_output=False,   # stream directly to terminal
    text=True,
)
sys.exit(result.returncode)
