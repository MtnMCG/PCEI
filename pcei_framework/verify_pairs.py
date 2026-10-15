import json
from pathlib import Path

db_path = Path("data/processed/training_database.jsonl")
found = 0

print("Scanning for Audio Pairs...")
with open(db_path, "r") as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            # Find records with an audio path that are not our custom cache
            if record.get("audio_path") and not record.get("audio_path").startswith("audio/"):
                if "librispeech" in record.get("audio_path", "").lower():
                    print(f"\n[Verified Pair {found+1}]")
                    print(f"🎤 Waveform: {record.get('audio_path')}")
                    print(f"📝 Transcript: \"{record.get('text')}\"")
                    found += 1
                    if found >= 3:
                        break
print(f"\nScan complete.")
