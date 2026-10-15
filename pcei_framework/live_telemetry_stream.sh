#!/usr/bin/env bash
# Continuous Live Telemetry & 10k-Step Checkpoint Sync from RTX 4090 to MacBook
REMOTE_HOST="${REMOTE_HOST:-<REMOTE_GPU_IP>}"
REMOTE_PORT="${REMOTE_PORT:-22}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")" && pwd)}"

LAST_SYNC_STEP=0
CHECKPOINT_INTERVAL=10000

echo "[$(date)] Telemetry & Checkpoint Daemon started."

while true; do
    # 1. Sync lightweight logs, plots, metrics every 60s
    rsync -aq --exclude="*_step*.json" -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
        root@$REMOTE_HOST:/workspace/pcei/src/logs/ \
        "$LOCAL_DIR/src/logs/" 2>/dev/null || true
    
    rsync -aq -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
        root@$REMOTE_HOST:/workspace/train.log \
        "$LOCAL_DIR/cloud_train.log" 2>/dev/null || true

    # 2. Check current step from newest loss summary
    NEWEST_CSV=$(ls -t "$LOCAL_DIR/src/logs/"*/loss_summary.csv 2>/dev/null | head -1)
    if [ -n "$NEWEST_CSV" ]; then
        RAW_STEP=$(tail -1 "$NEWEST_CSV" 2>/dev/null | cut -d',' -f1 | tr -d ' ')
        if [[ "$RAW_STEP" =~ ^[0-9]+$ ]]; then
            CURRENT_STEP=$RAW_STEP
        else
            CURRENT_STEP=0
        fi
        
        # Check if crossed 10k boundary
        if [ "$LAST_SYNC_STEP" -eq 0 ]; then
            LAST_SYNC_STEP=$(( (CURRENT_STEP / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL ))
        fi
        STEP_DIFF=$((CURRENT_STEP - LAST_SYNC_STEP))
        if [ "$STEP_DIFF" -ge "$CHECKPOINT_INTERVAL" ]; then
            echo "[$(date)] Crossed 10k step milestone (Step: $CURRENT_STEP). Syncing checkpoint..."
            rsync -avz -e "ssh -i $KEY_PATH -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
                root@$REMOTE_HOST:/workspace/pcei/checkpoints/pcei_750m_supercharged.pt \
                "$LOCAL_DIR/checkpoints/pcei_750m_supercharged.pt"
            
            # Save milestone snapshot
            cp "$LOCAL_DIR/checkpoints/pcei_750m_supercharged.pt" \
               "$LOCAL_DIR/checkpoints/pcei_750m_milestone_${CURRENT_STEP}.pt" 2>/dev/null || true
            
            LAST_SYNC_STEP=$CURRENT_STEP
            echo "[$(date)] Checkpoint synced & milestone_${CURRENT_STEP}.pt saved!"
        fi
    fi
        
    sleep 60
done
