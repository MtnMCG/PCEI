#!/usr/bin/env python3
"""
pcei_framework/tools/acquire_conversational_diversity.py
Downloads high-diversity, multi-turn dialogues from UltraChat and SmolTalk via Hugging Face API
to inject authentic linguistic diversity, adult conversational syntax, and eliminate template repetition.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_ultrachat(total_rows: int = 1500):
    dst = DATA_DIR / "ultrachat_conversations.jsonl"
    print(f"[UltraChat] Fetching {total_rows} multi-turn conversational records...")
    
    records = []
    chunk_size = 100
    for offset in range(0, total_rows, chunk_size):
        url = f"https://datasets-server.huggingface.co/rows?dataset=openbmb%2FUltraChat&config=default&split=train&offset={offset}&length={chunk_size}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = json.loads(r.read().decode("utf-8"))
                for row_item in payload.get("rows", []):
                    row = row_item.get("row", {})
                    data_dialogue = row.get("data", [])
                    if len(data_dialogue) >= 2:
                        # Format as clean USER / ASSISTANT multi-turn dialogue
                        turns = []
                        for i, utterance in enumerate(data_dialogue):
                            role = "USER" if i % 2 == 0 else "ASSISTANT"
                            turns.append(f"{role}: {utterance.strip()}")
                        records.append({"text": "\n".join(turns)})
            print(f"  • Ingested {len(records)} UltraChat dialogues...", flush=True)
        except Exception as e:
            print(f"  [Warning] Batch at offset {offset} failed: {e}")
            break
            
    with open(dst, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[UltraChat] Successfully saved {len(records)} records to {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)")


def fetch_smoltalk(total_rows: int = 1500):
    dst = DATA_DIR / "smoltalk_dialogues.jsonl"
    print(f"[SmolTalk] Fetching {total_rows} reasoning and conversational records...")
    
    records = []
    chunk_size = 100
    for offset in range(0, total_rows, chunk_size):
        url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceTB%2Fsmoltalk&config=all&split=train&offset={offset}&length={chunk_size}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = json.loads(r.read().decode("utf-8"))
                for row_item in payload.get("rows", []):
                    row = row_item.get("row", {})
                    messages = row.get("messages", [])
                    if messages:
                        turns = []
                        for msg in messages:
                            role = "USER" if msg.get("role", "").lower() in ["user", "human"] else "ASSISTANT"
                            content = str(msg.get("content", "")).strip()
                            if content:
                                turns.append(f"{role}: {content}")
                        if turns:
                            records.append({"text": "\n".join(turns)})
            print(f"  • Ingested {len(records)} SmolTalk dialogues...", flush=True)
        except Exception as e:
            print(f"  [Warning] Batch at offset {offset} failed: {e}")
            break
            
    with open(dst, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[SmolTalk] Successfully saved {len(records)} records to {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    fetch_ultrachat(1500)
    fetch_smoltalk(1500)
