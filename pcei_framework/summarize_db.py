import json
from pathlib import Path

db_path = Path("data/processed/training_database.jsonl")
total_records = 0
type_counts = {}

with open(db_path, "r") as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            total_records += 1
            rtype = record.get("type", "unknown")
            type_counts[rtype] = type_counts.get(rtype, 0) + 1

print(f"Total Base Records in DB: {total_records:,}")
print("--- Breakdown by Type ---")
for k, v in type_counts.items():
    print(f"  {k}: {v:,}")
