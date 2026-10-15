#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/../pcei_framework" && pwd)"
TARGET_DIR="$SCRIPT_DIR/PCEI_Docker/pcei_framework"

echo "Creating pristine export directory..."
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "Copying source code..."
cp -r "$SOURCE_DIR/src" "$TARGET_DIR/"
cp -r "$SOURCE_DIR/scripts" "$TARGET_DIR/"
cp -r "$SOURCE_DIR/tools" "$TARGET_DIR/"
cp "$SOURCE_DIR/COMMANDS.md" "$TARGET_DIR/"

echo "Copying checkpoints (excluding massive backups)..."
mkdir -p "$TARGET_DIR/checkpoints"
# Only copy the essential 500m checkpoint
cp "$SOURCE_DIR/checkpoints/pcei_500m.pt" "$TARGET_DIR/checkpoints/" || echo "pcei_500m.pt not found, skipping."

echo "Creating empty data directories..."
mkdir -p "$TARGET_DIR/data/raw/text"
mkdir -p "$TARGET_DIR/data/raw/image"
mkdir -p "$TARGET_DIR/data/raw/audio"
mkdir -p "$TARGET_DIR/data/processed"
mkdir -p "$TARGET_DIR/data/high_quality"

echo "Done. Codebase exported cleanly without venv or __pycache__."
