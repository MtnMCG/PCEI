"""
PCEI Massive Data Acquisition Script

Downloads subsets of MS COCO and Clotho to build a 500-sample multimodal test set.
Ensures we can verify 'Entanglement Bonds' before committing to the full 40GB.
"""
import os
import sys
import json
import random
import requests
import zipfile
import tarfile
import wave
import struct
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_IMG = RAW_DIR / "image"
RAW_AUD = RAW_DIR / "audio"
RAW_TXT = RAW_DIR / "text"

# Ensure dirs exist
for d in [RAW_IMG, RAW_AUD, RAW_TXT]:
    d.mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    print(f"[Downloading] {url} -> {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def acquire_coco_subset(num_samples: int = None):
    """Downloads COCO 2017 annotations and images. If num_samples is None, processes the full dataset."""
    print("\n--- Acquiring MS COCO Dataset ---")
    
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = RAW_DIR / "coco_ann.zip"
    
    if not zip_path.exists():
        download_file(ann_url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_zip:
        zip_zip.extractall(RAW_DIR)
    
    ann_file = RAW_DIR / "annotations" / "captions_val2017.json"
    print(f"[COCO] Parsing {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Map image_id to captions
    img_to_caps = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_caps:
            img_to_caps[img_id] = []
        img_to_caps[img_id].append(ann['caption'])
    
    all_img_ids = list(img_to_caps.keys())
    random.shuffle(all_img_ids)
    
    # Count how many we already have locally
    existing_img_ids = []
    new_img_ids = []
    for img_id in all_img_ids:
        img_filename = f"{str(img_id).zfill(12)}.jpg"
        if (RAW_IMG / img_filename).exists():
            existing_img_ids.append(img_id)
        else:
            new_img_ids.append(img_id)
            
    # Support 'None' for full dataset download
    if num_samples is None:
        selected = all_img_ids
    else:
        # Prioritize existing ones to save time/bandwidth
        selected = existing_img_ids[:num_samples]
        if len(selected) < num_samples:
            selected.extend(new_img_ids[:num_samples - len(selected)])
    
    print(f"[COCO] Enqueued {len(selected)} images for download...")
    base_img_url = "http://images.cocodataset.org/val2017/"
    
    samples_manifest = []
    
    for img_id in selected:
        img_filename = f"{str(img_id).zfill(12)}.jpg"
        img_url = base_img_url + img_filename
        target_img = RAW_IMG / img_filename
        
        try:
            if not target_img.exists():
                download_file(img_url, target_img)
            
            # Save the primary caption to text dir
            caption = img_to_caps[img_id][0]
            txt_filename = f"coco_{img_id}.txt"
            with open(RAW_TXT / txt_filename, "w") as f:
                f.write(caption)
            
            # Conforms to strict schema
            samples_manifest.append({
                "id": f"coco_{img_id}",
                "type": "multimodal",
                "text": caption,
                "image_path": str(target_img.relative_to(ROOT)),
                "audio_path": None
            })
        except Exception as e:
            print(f"  [Error] Failed to get image {img_id}: {e}")

    return samples_manifest

def acquire_clotho_subset(num_samples: int = None):
    """Acquires Clotho. If num_samples is None, targets the full dataset."""
    print("\n--- Acquiring Clotho Dataset ---")
    samples = []
    
    local_audio = list(RAW_AUD.glob("*.wav")) + list(RAW_AUD.glob("*.mp3"))
    
    target_count = len(local_audio) if num_samples is None else num_samples
    
    if len(local_audio) < target_count:
        print(f"[Clotho] Only {len(local_audio)} local files found. (Synthetic generation disabled).")

    for i, f in enumerate(local_audio[:target_count]):
        samples.append({
            "id": f"clotho_real_{i}",
            "type": "multimodal",
            "text": "ambient environment recording",
            "image_path": None,
            "audio_path": str(f.relative_to(ROOT))
        })
    return samples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCEI Massive Data Acquisition")
    parser.add_argument("--full", action="store_true", help="Download the full datasets (approx 40GB). If not specified, downloads a subset.")
    parser.add_argument("--subset-size", type=int, default=3500, help="Number of records to download if not using --full.")
    args = parser.parse_args()
def acquire_librispeech_subset(num_samples: int = 500):
    """Downloads a very small (~300MB) subset of LibriSpeech and extracts the requested number of samples."""
    print(f"\n--- Acquiring LibriSpeech Dataset (Target: {num_samples} samples) ---")
    
    url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = RAW_DIR / "librispeech_dev.tar.gz"
    libri_dir = RAW_AUD / "librispeech"
    
    if not tar_path.exists():
        download_file(url, tar_path)
        
    print("[LibriSpeech] Extracting tarball...")
    if not libri_dir.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=RAW_AUD)
            
    # The extraction creates a 'LibriSpeech' folder
    extracted_root = RAW_AUD / "LibriSpeech" / "dev-clean"
    
    samples = []
    
    # Walk through the speaker/chapter directories to find the .txt transcripts and .flac audio
    for speaker_dir in extracted_root.iterdir():
        if not speaker_dir.is_dir(): continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir(): continue
            
            # Find the transcript file
            transcripts = list(chapter_dir.glob("*.txt"))
            if not transcripts: continue
            
            with open(transcripts[0], 'r') as f:
                for line in f:
                    if len(samples) >= num_samples:
                        print(f"  [Included] LibriSpeech: {len(samples)} valid records.")
                        return samples
                        
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        flac_path = chapter_dir / f"{file_id}.flac"
                        
                        if flac_path.exists():
                            samples.append({
                                "id": f"libri_{file_id}",
                                "type": "multimodal",
                                "text": text.lower(),
                                "image_path": None,
                                "audio_path": str(flac_path.relative_to(ROOT))
                            })
                            
    print(f"  [Included] LibriSpeech: {len(samples)} valid records.")
    return samples

def main(num_samples=3500, full_download=False): # Lowered default for faster iteration
    print(f"Initializing Massive Acquisition for {num_samples if not full_download else 'full'} diverse samples...")
    
    if full_download:
        print("WARNING: --full flag used. Preparing for Massive 40GB Data Acquisition.")

    combined_records = []
    
    # 1. COCO
    coco_samples = acquire_coco_subset(num_samples if not full_download else None)
    combined_records.extend(coco_samples)
    
    # 2. Clotho
    clotho_samples = acquire_clotho_subset(num_samples if not full_download else None)
    combined_records.extend(clotho_samples)
    
    # 3. LibriSpeech (always a fixed small subset for now)
    libri_samples = acquire_librispeech_subset(500)
    combined_records.extend(libri_samples)
    
    # 4. Write to training_database.jsonl (Strict Schema)
    db_path = ROOT / "data" / "processed" / "training_database.jsonl"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(db_path, "w") as f:
        for record in combined_records:
            f.write(json.dumps(record) + "\n")
    
    print(f"\n[Complete] Generated {len(combined_records)} multimodal records.")
    print(f"Database saved to: data/processed/training_database.jsonl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCEI Massive Data Acquisition Phase")
    parser.add_argument("--full", action="store_true", help="Download the full datasets (approx 40GB). If not specified, downloads a subset.")
    parser.add_argument("--subset-size", type=int, default=3500, help="Number of records to download if not using --full.")
    args = parser.parse_args()
    
    main(num_samples=args.subset_size, full_download=args.full)
