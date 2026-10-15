#!/usr/bin/env bash
# Synchronize checkpoints and metrics logs from RTX 4090 cloud instance to local Mac
REMOTE_HOST="${REMOTE_HOST:-<REMOTE_GPU_IP>}"
REMOTE_PORT="${REMOTE_PORT:-22}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "=== Pulling Latest Telemetry & Checkpoints from Cloud ==="
# 1. Sync metrics logs & plots
rsync -avz -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
    root@$REMOTE_HOST:/workspace/pcei/src/logs/ \
    "$LOCAL_DIR/src/logs/"

# 2. Sync train.log
rsync -avz -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
    root@$REMOTE_HOST:/workspace/train.log \
    "$LOCAL_DIR/cloud_train.log"

# 3. Fast Sync of New Wired Model (Lightweight 1.7GB Inference Checkpoint, no CPU compression choke)
echo "Downloading Phase 1 & 3 Wired Foundation Model (1.7 GB)..."
rsync -av --progress -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
    root@$REMOTE_HOST:/workspace/pcei/checkpoints/pcei_750m_phase1_3_inference.pt \
    "$LOCAL_DIR/checkpoints/pcei_750m_phase1_3_inference.pt"

echo "=== Sync Complete at $(date) ==="
