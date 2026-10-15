import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# --- CONFIG ---
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VIDEO_CACHE = PROCESSED_DIR / "video_cache"
CONGRUENT_FILE = ROOT / "data" / "synthetic" / "congruent_concepts.jsonl"
OUTPUT_DB = PROCESSED_DIR / "training_database.jsonl"
LABEL_CACHE = PROCESSED_DIR / "label_cache.json"

# --- INIT ---
VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
(VIDEO_CACHE / "frames").mkdir(exist_ok=True)
(VIDEO_CACHE / "audio").mkdir(exist_ok=True)

def load_congruent_concepts() -> List[Dict]:
    concepts = []
    if CONGRUENT_FILE.exists():
        with open(CONGRUENT_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("tag") == "custom":
                            text = record["text"]
                            # Pre-calculate keywords for faster matching
                            keywords = set(k.lower() for k in text.split() if len(k) > 3)
                            concepts.append({"text": text, "keywords": keywords})
                    except: pass
    return concepts

def match_label(filename: str, concepts: List[Dict]) -> str:
    """Optimized matching using keyword sets."""
    fname_lower = filename.lower()
    
    # Try exact stem match first
    stem = Path(filename).stem.lower()
    
    for concept_data in concepts:
        text = concept_data["text"]
        keywords = concept_data["keywords"]
        
        # If any significant keyword from concept is in filename, match it
        for k in keywords:
            if k in fname_lower:
                return text
            
    return f"unlabeled signal: {filename}"

def process_video(video_path: Path, fps: float = 2.0, max_frames: int = 16):
    """Explode video into persistent frames and audio for high-quality multimodal training."""
    video_id = video_path.stem
    frame_dir = VIDEO_CACHE / "frames" / video_id
    frame_dir.mkdir(exist_ok=True)
    audio_path = VIDEO_CACHE / "audio" / f"{video_id}.wav"
    
    # Check if ffmpeg is available
    ffmpeg_found = shutil.which("ffmpeg") is not None
    if not ffmpeg_found:
        print(f"  [Warning] ffmpeg not found. Skipping video processing for {video_id}.")
        return None, None

    # 1. Extract Frames
    if not any(frame_dir.glob("*.png")):
        print(f"  [Video] Extracting frames: {video_id}")
        try:
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-frames:v", str(max_frames),
                "-loglevel", "error",
                str(frame_dir / "frame_%04d.png")
            ], check=True)
        except Exception as e:
            print(f"  [Error] Failed to extract frames for {video_id}: {e}")
    
    # 2. Extract Audio
    if not audio_path.exists():
        print(f"  [Video] Extracting audio: {video_id}")
        try:
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-loglevel", "error",
                str(audio_path)
            ], check=True)
        except Exception as e:
            print(f"  [Error] Failed to extract audio for {video_id}: {e}")
        
    return frame_dir, audio_path

def audit_dataset():
    print("═" * 40)
    print("  PCEI Dataset Integrity Audit")
    print("═" * 40)
    
    concepts = load_congruent_concepts()
    print(f"[Audit] Loaded {len(concepts)} custom concepts for matching.")
    
    records = []
    
    # Scan Images
    print("[Audit] Scanning images...")
    img_dir = RAW_DIR / "image"
    if img_dir.exists():
        for f in img_dir.glob("*.*"):
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']:
                label = match_label(f.name, concepts)
                records.append({
                    "id": f.stem,
                    "type": "multimodal",
                    "text": label,
                    "image_path": str(f.relative_to(ROOT)),
                    "audio_path": None
                })
                
    # Scan Audio
    print("[Audit] Scanning audio...")
    aud_dir = RAW_DIR / "audio"
    if aud_dir.exists():
        for f in aud_dir.glob("*.*"):
            if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
                label = match_label(f.name, concepts)
                records.append({
                    "id": f.stem,
                    "type": "multimodal",
                    "text": label,
                    "image_path": None,
                    "audio_path": str(f.relative_to(ROOT))
                })
                
    # Scan Video
    print("[Audit] Scanning videos...")
    vid_dir = RAW_DIR / "video"
    if vid_dir.exists():
        for f in vid_dir.glob("*.*"):
            if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.webm']:
                label = match_label(f.name, concepts)
                frame_dir, audio_path = process_video(f)
                if frame_dir and audio_path:
                    records.append({
                        "id": f.stem,
                        "type": "video_multimodal",
                        "text": label,
                        "image_path": str(frame_dir.relative_to(ROOT)), # Path to directory
                        "audio_path": str(audio_path.relative_to(ROOT))
                    })
                else:
                    # Fallback to simple video record if ffmpeg failed/missing
                    records.append({
                        "id": f.stem,
                        "type": "custom_video", # Legacy type
                        "text": label,
                        "image_path": None,
                        "audio_path": str(f.relative_to(ROOT))
                    })

    # Load existing database to preserve COCO data
    print(f"[Audit] Preserving existing records from {OUTPUT_DB.name}")
    existing_ids = {r["id"] for r in records}
    if OUTPUT_DB.exists():
        with open(OUTPUT_DB, 'r') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["id"] not in existing_ids:
                        records.append(r)
    
    # Write updated DB
    with open(OUTPUT_DB, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"═" * 40)
    print(f"[Audit Complete] Total records in training_database.jsonl: {len(records)}")
    print("═" * 40)

if __name__ == "__main__":
    audit_dataset()
