#!/bin/bash
# wait_and_train.sh
# Polls until the dataset download finishes, then launches training

LOGDIR="/Users/emerson/Desktop/PCEI /pcei_framework/logs"
DATAFILE="/Users/emerson/Desktop/PCEI /pcei_framework/data/synthetic/anthropic_claude.jsonl"
VENV="/Users/emerson/Desktop/PCEI /pcei_framework/venv/bin/python"
SCRIPT="/Users/emerson/Desktop/PCEI /pcei_framework/tools/instruction_tuner.py"

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
