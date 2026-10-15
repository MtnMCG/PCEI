#!/usr/bin/env python3
"""
Acquire Conversational Diversity Pack:
1. Databricks Dolly-15k (15,011 curated human instruction-response pairs) ~7.4 MB
2. OpenAssistant OASST1 (88,838 human conversational turns) ~39.7 MB
3. Reddit Title-Body Corpus (7,011,989 threads across thousands of subreddits) ~3.18 GB (10 Parquet chunks)

Saves into: pcei_framework/data/conversational/
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "conversational"
REDDIT_DIR = DATA_DIR / "reddit"

DATASETS = {
    "dolly_15k": {
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "path": DATA_DIR / "dolly_15k.parquet",
        "expected_mb": 7.39
    },
    "oasst1_train": {
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "path": DATA_DIR / "oasst1_train.parquet",
        "expected_mb": 37.66
    },
    "oasst1_val": {
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        "path": DATA_DIR / "oasst1_val.parquet",
        "expected_mb": 2.01
    }
}

REDDIT_BASE_URL = "https://huggingface.co/datasets/sentence-transformers/reddit-title-body/resolve/refs%2Fconvert%2Fparquet/default/partial-train/{:04d}.parquet"


def download_file(url: str, dest_path: Path, desc: str):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[{desc}] Already exists ({dest_path.stat().st_size / (1024*1024):.2f} MB), skipping.")
        return

    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PCEI; DataAcquisition)"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 512  # 512 KB
            last_print = t0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_size:
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    speed_mb = (downloaded / (1024 * 1024)) / (now - t0 + 1e-6)
                    print(f"[{desc}] {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed_mb:.2f} MB/s", end="\r", flush=True)
                    last_print = now

        temp_path.rename(dest_path)
        dur = time.time() - t0
        print(f"\n[{desc}] Completed {dest_path.stat().st_size / (1024*1024):.2f} MB in {dur:.1f}s")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def main():
    print("======================================================================")
    print("  PCEI CONVERSATIONAL CORPUS ACQUISITION (DOLLY + OASST + 3GB REDDIT) ")
    print("======================================================================")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REDDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download Dolly 15k
    print("\n1/3. Acquiring Databricks Dolly-15k...")
    download_file(DATASETS["dolly_15k"]["url"], DATASETS["dolly_15k"]["path"], "Dolly-15k")

    # 2. Download OpenAssistant
    print("\n2/3. Acquiring OpenAssistant OASST1...")
    download_file(DATASETS["oasst1_train"]["url"], DATASETS["oasst1_train"]["path"], "OASST1-Train")
    download_file(DATASETS["oasst1_val"]["url"], DATASETS["oasst1_val"]["path"], "OASST1-Val")

    # 3. Download Reddit Parquet Files (10 files, ~3.18 GB)
    print("\n3/3. Acquiring Full 3.18 GB Reddit Title-Body Corpus (10 chunks)...")
    for i in range(10):
        url = REDDIT_BASE_URL.format(i)
        dest = REDDIT_DIR / f"{i:04d}.parquet"
        download_file(url, dest, f"Reddit Part {i+1}/10")

    print("\n======================================================================")
    print("  ALL CONVERSATIONAL DATASETS DOWNLOADED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()
