#!/usr/bin/env python3
"""Publishes the rich Model Card to MtnMCG/PCEI-Text on Hugging Face."""
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "MtnMCG/PCEI-Text"
CARD_PATH = Path(__file__).resolve().parent.parent / "docs" / "HF_MODEL_CARD_PCEI_TEXT.md"

if not CARD_PATH.exists():
    print(f"Error: Card file not found at {CARD_PATH}")
    sys.exit(1)

print(f"Uploading {CARD_PATH} to {REPO_ID} as README.md...")
api = HfApi()
api.upload_file(
    path_or_fileobj=str(CARD_PATH),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model"
)
print(f"Successfully published Model Card to https://huggingface.co/{REPO_ID}!")
