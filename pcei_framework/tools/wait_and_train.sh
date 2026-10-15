#!/bin/bash
# wait_and_train.sh
# Polls until the dataset download finishes, then launches training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGDIR="$FRAMEWORK_DIR/logs"
DATAFILE="$FRAMEWORK_DIR/data/synthetic/anthropic_claude.jsonl"
VENV="$FRAMEWORK_DIR/venv/bin/python"
SCRIPT="$SCRIPT_DIR/instruction_tuner.py"

echo "[launcher] Waiting for dataset download to complete..."

while true; do
    if [ -f "$DATAFILE" ]; then
        # Check if download process is still running
        if ! pgrep -f "download_claude_dataset.py" > /dev/null 2>&1; then
            LINES=$(wc -l < "$DATAFILE")
            echo "[launcher] Download complete. $LINES records on disk. Launching training..."
            break
        fi
    fi
    sleep 30
    echo "[launcher] Still downloading... $(date)"
done

# Launch caffeinated training
exec caffeinate -i "$VENV" "$SCRIPT" \
    --mode anthropic \
    --num-ollama 20000 \
    --metabolic-flush \
    --grad-accum-steps 8
