#!/usr/bin/env bash
set -e

REMOTE_HOST="${1:-ssh1.vast.ai}"
REMOTE_PORT="${2:-22}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SSH_CMD="ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no"

echo "================================================================"
echo "  PCEI PULL ALIGNED MODEL FROM CLOUD"
echo "  Date: $(date)"
echo "================================================================"

# 1. Sync training logs
echo "[Pull] Syncing logs..."
rsync -avz -e "$SSH_CMD" \
    root@$REMOTE_HOST:/workspace/alignment_training.log \
    "$LOCAL_DIR/cloud_alignment_training.log" || true

rsync -avz -e "$SSH_CMD" \
    root@$REMOTE_HOST:/workspace/targeted_annealing.log \
    "$LOCAL_DIR/cloud_targeted_annealing.log" || true

# 2. Sync aligned weights
echo "[Pull] Downloading pcei_750m_aligned_v2.pt..."
rsync -av --progress -e "$SSH_CMD" \
    root@$REMOTE_HOST:/workspace/pcei/checkpoints/pcei_750m_aligned_v2.pt \
    "$LOCAL_DIR/checkpoints/pcei_750m_aligned_v2.pt"

echo "[Pull] Downloading pcei_750m_reasoning_master.pt..."
rsync -av --progress -e "$SSH_CMD" \
    root@$REMOTE_HOST:/workspace/pcei/checkpoints/pcei_750m_reasoning_master.pt \
    "$LOCAL_DIR/checkpoints/pcei_750m_reasoning_master.pt"

echo "================================================================"
echo "  PULL COMPLETE AT $(date)"
echo "================================================================"
