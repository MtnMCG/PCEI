"""
build_manifest.py — Unified Multimodal Manifest Builder

Scans:
  - data/raw/audio/LibriSpeech/  (FLAC files + .trans.txt transcripts)
  - data/raw/image/              (COCO JPEG files + existing manifest captions)

Outputs:
  - data/massive_data_test_manifest.json  (unified, replaces old image-only version)

Usage:
    python tools/build_manifest.py
    python tools/build_manifest.py --max-audio 2000 --max-image 500
"""

import json
import argparse
import random
from pathlib import Path

BASE = Path(__file__).parent.parent


def build_audio_records(librispeech_root: Path, max_records: int) -> list:
    """
    Walk LibriSpeech directory tree.
    Each chapter dir has a .trans.txt: '<uttid> TRANSCRIPT TEXT'
    Each utterance has a matching .flac file in the same dir.
    Returns list of manifest records with modality='audio'.
    """
    records = []
    trans_files = sorted(librispeech_root.rglob("*.trans.txt"))
    print(f"  Found {len(trans_files)} LibriSpeech transcript files")

    for trans_path in trans_files:
        chapter_dir = trans_path.parent
        with open(trans_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                # Transcript is all-caps in LibriSpeech — title-case it
                transcript = transcript.capitalize()

                # FLAC file is in same dir
                flac_path = chapter_dir / f"{utt_id}.flac"
                if not flac_path.exists():
                    continue

                records.append({
                    "modality": "audio",
                    "file": str(flac_path.relative_to(BASE)),
                    "caption": transcript,
                    "speaker_id": utt_id.split("-")[0],
                })

                if len(records) >= max_records:
                    return records

    return records


def build_image_records(image_dir: Path, existing_manifest: Path, max_records: int) -> list:
    """
    Load captioned COCO images from existing manifest first (quality data),
    then fill remaining slots with uncaptioned images from image_dir.
    Returns list of manifest records with modality='image'.
    """
    records = []

    # 1. Load existing captioned records first
    if existing_manifest.exists():
        with open(existing_manifest) as f:
            existing = json.load(f)
        captioned = [r for r in existing if r.get("modality") == "image" and
                     r.get("caption") and not r["caption"].startswith("unlabeled")]
        records.extend(captioned)
        print(f"  Loaded {len(captioned)} captioned image records from existing manifest")

    # 2. Scan for additional images not already in records
    existing_files = {r["file"] for r in records}
    all_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    random.shuffle(all_images)

    added = 0
    for img_path in all_images:
        rel = str(img_path.relative_to(BASE))
        if rel in existing_files:
            continue
        records.append({
            "modality": "image",
            "file": rel,
            "caption": f"an image",  # generic — gets refined by captioning training
        })
        added += 1
        if len(records) >= max_records:
            break

    print(f"  Added {added} additional uncaptioned images")
    return records[:max_records]


def main():
    parser = argparse.ArgumentParser(description="Build unified PCEI multimodal manifest")
    parser.add_argument("--max-audio", type=int, default=2500,
                        help="Max audio records to include (default: 2500 = ~full LibriSpeech dev-clean)")
    parser.add_argument("--max-image", type=int, default=1000,
                        help="Max image records to include (default: 1000)")
    parser.add_argument("--output", type=str,
                        default="data/massive_data_test_manifest.json",
                        help="Output manifest path")
    args = parser.parse_args()

    output_path = BASE / args.output

    print("=" * 55)
    print("  PCEI Manifest Builder")
    print("=" * 55)

    # ── Audio ──────────────────────────────────────────────
    librispeech_root = BASE / "data" / "raw" / "audio" / "LibriSpeech"
    print(f"\n[Audio] Scanning {librispeech_root}...")
    if librispeech_root.exists():
        audio_records = build_audio_records(librispeech_root, args.max_audio)
        print(f"  → {len(audio_records)} audio records")
    else:
        print("  LibriSpeech not found, skipping audio")
        audio_records = []

    # ── Images ─────────────────────────────────────────────
    image_dir = BASE / "data" / "raw" / "image"
    existing_manifest = output_path  # reuse captions from current file if present
    print(f"\n[Image] Scanning {image_dir}...")
    if image_dir.exists():
        image_records = build_image_records(image_dir, existing_manifest, args.max_image)
        print(f"  → {len(image_records)} image records")
    else:
        print("  Image dir not found, skipping images")
        image_records = []

    # ── Combine & shuffle ──────────────────────────────────
    all_records = image_records + audio_records
    random.shuffle(all_records)

    # ── Write ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # Summary
    n_img = sum(1 for r in all_records if r["modality"] == "image")
    n_aud = sum(1 for r in all_records if r["modality"] == "audio")
    captioned_img = sum(1 for r in all_records if r["modality"] == "image" and r["caption"] != "an image")

    print(f"\n{'=' * 55}")
    print(f"  Manifest written: {output_path}")
    print(f"  Total records:    {len(all_records)}")
    print(f"  Image records:    {n_img}  ({captioned_img} with real captions)")
    print(f"  Audio records:    {n_aud}  (all with transcripts)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
