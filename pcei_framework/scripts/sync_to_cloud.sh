#!/usr/bin/env bash
set -e

REMOTE_HOST="${1:-ssh1.vast.ai}"
REMOTE_PORT="${2:-22}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SSH_CMD="ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no"

echo "================================================================"
echo "  PCEI LEAN CLOUD SYNC -> $REMOTE_HOST:$REMOTE_PORT"
echo "  Date: $(date)"
echo "================================================================"

# 1. Create directory structure on remote
echo "[Sync] Creating remote directories..."
$SSH_CMD root@$REMOTE_HOST "mkdir -p /workspace/pcei/src /workspace/pcei/tools /workspace/pcei/scripts /workspace/pcei/data/curriculum /workspace/pcei/data/synthetic /workspace/pcei/checkpoints /workspace/pcei/tiktoken_cache"

# 2. Sync src (excluding logs and pycache)
echo "[Sync] Syncing src/..."
rsync -avz --exclude='logs' --exclude='__pycache__' -e "$SSH_CMD"     "$LOCAL_DIR/src/" root@$REMOTE_HOST:/workspace/pcei/src/

# 3. Sync tools
echo "[Sync] Syncing tools/..."
rsync -avz --exclude='__pycache__' -e "$SSH_CMD"     "$LOCAL_DIR/tools/" root@$REMOTE_HOST:/workspace/pcei/tools/

# 4. Sync scripts
echo "[Sync] Syncing scripts/..."
rsync -avz --exclude='__pycache__' -e "$SSH_CMD"     "$LOCAL_DIR/scripts/" root@$REMOTE_HOST:/workspace/pcei/scripts/

# 5. Sync tiktoken cache
if [ -d "$LOCAL_DIR/tiktoken_cache" ]; then
    echo "[Sync] Syncing tiktoken_cache/..."
    rsync -avz -e "$SSH_CMD"         "$LOCAL_DIR/tiktoken_cache/" root@$REMOTE_HOST:/workspace/pcei/tiktoken_cache/
fi

# 6. Sync curated clean datasets (< 60MB total)
echo "[Sync] Syncing conversational reasoning datasets..."
rsync -avz -e "$SSH_CMD"     "$LOCAL_DIR/data/curriculum/pcei_foundation_teacher.jsonl"     root@$REMOTE_HOST:/workspace/pcei/data/curriculum/

rsync -avz -e "$SSH_CMD"     "$LOCAL_DIR/data/synthetic/anthropic_claude.jsonl"     "$LOCAL_DIR/data/synthetic/beast_reasoning.jsonl"     "$LOCAL_DIR/data/synthetic/congruent_concepts.jsonl"     root@$REMOTE_HOST:/workspace/pcei/data/synthetic/

# 7. Sync base checkpoint (1.7 GB)
echo "[Sync] Syncing pcei_750m_sanitized_base.pt (1.7 GB)..."
rsync -av --progress -e "$SSH_CMD"     "$LOCAL_DIR/checkpoints/pcei_750m_sanitized_base.pt"     root@$REMOTE_HOST:/workspace/pcei/checkpoints/pcei_750m_sanitized_base.pt

echo "================================================================"
echo "  SYNC COMPLETE AT $(date)"
echo "================================================================"
