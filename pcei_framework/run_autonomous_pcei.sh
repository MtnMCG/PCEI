#!/usr/bin/env bash
# ==============================================================================
# PCEI 750M Autonomous Resilient Training Harness
# Runs indefinitely, auto-saves checkpoints, and auto-revives on any crash/OOM
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=================================================================="
echo "   PCEI 750M — Autonomous Crash-Resilient Training Loop"
echo "   Running in External Detached Terminal (Independent of IDE/Agent)"
echo "=================================================================="

source venv/bin/activate

CHECKPOINT="pcei_750m.pt"
LOG_FILE="logs/autonomous_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Target Checkpoint: $CHECKPOINT"
echo "Logging output to: $LOG_FILE"
echo ""

CRASH_COUNT=0

export PYTHONUNBUFFERED=1

while true; do
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching / Resuming PCEI Training Session..."
    echo "=================================================================="
    
    # Prevent macOS sleep during compute with caffeinate
    caffeinate -i -s python -u scripts/train.py \
        --checkpoint-name "$CHECKPOINT" \
        --resume \
        --high-quality-only \
        --lr 2e-4 \
        --grad-accum-steps 8 \
        --metabolic-flush \
        --bootstrap-steps 5000 \
        --integration-steps 2500 \
        --open-steps 2500 \
        2>&1 | tee -a "$LOG_FILE"
        
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] Full training run completed successfully!"
        echo "Running evaluation benchmark..."
        python tools/benchmark.py --checkpoint-name "$CHECKPOINT" | tee -a "$LOG_FILE"
        break
    else
        CRASH_COUNT=$((CRASH_COUNT + 1))
        echo ""
        echo "⚠️ [$(date '+%Y-%m-%d %H:%M:%S')] Process exited with code $EXIT_CODE (Crash count: $CRASH_COUNT)."
        echo "Cooling down MPS unified memory for 10 seconds before auto-revival..."
        sleep 10
        echo "Reviving PCEI and resuming from last safe checkpoint ($CHECKPOINT)..."
    fi
done
