import json
import sys
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "processed" / "training_database.jsonl"

def schema_check():
    print(f"\n--- Schema Check: {DB_PATH.name} ---")
    if not DB_PATH.exists():
        print("[Error] Database not found.")
        sys.exit(1)
        
    records = []
    with open(DB_PATH, "r") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                record = json.loads(line)
                
                # Check for required keys
                missing = [k for k in ('id', 'type', 'text', 'image_path', 'audio_path') if k not in record]
                if missing:
                    print(f"  [Fail] Line {i+1} missing keys: {missing}")
                    sys.exit(1)
                
                # Check value types
                if not isinstance(record['id'], str):
                    print(f"  [Fail] Line {i+1} id must be string")
                    sys.exit(1)
                
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  [Fail] Corrupted JSON at line {i+1}: {e}")
                sys.exit(1)
                
    print(f"[Pass] Schema check successful for {len(records)} records.")
    return records

def ghost_walk(records, sample_size=100):
    print(f"\n--- Ghost Walk Path Verification ---")
    if len(records) < sample_size:
        sample_size = len(records)
        print(f"(Sampling all {sample_size} records)")
        
    samples = random.sample(records, sample_size)
    
    missing_files = 0
    for r in samples:
        img_ok = True
        aud_ok = True
        
        if r['image_path'] is not None:
            if not (ROOT / r['image_path']).exists():
                print(f"  [Missing] Image: {r['image_path']}")
                img_ok = False
        
        if r['audio_path'] is not None:
            if not (ROOT / r['audio_path']).exists():
                print(f"  [Missing] Audio: {r['audio_path']}")
                aud_ok = False
                
        if not img_ok or not aud_ok:
            missing_files += 1
            
    if missing_files > 0:
        print(f"[Fail] Ghost walk found {missing_files} missing files.")
        sys.exit(1)
    else:
        print(f"[Pass] Ghost walk verified paths for {sample_size} samples.")

if __name__ == "__main__":
    records = schema_check()
    ghost_walk(records)
