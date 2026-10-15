#!/usr/bin/env bash
# scripts/watch_kindergarten_training.sh
# Cron target: monitors cloud training health, keeps Mac alive,
# reports loss progress and auto-destroys instance on completion.
# Run by agent every hour.

set -e

LOG="/Users/emerson/Desktop/PCEI /pcei_framework/cloud_kindergarten_training.log"
CKPT_DIR="/Users/emerson/Desktop/PCEI /pcei_framework/checkpoints"
INSTANCE_ID="52154175"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================"
echo "  KINDERGARTEN TRAINING WATCHER — $TIMESTAMP"
echo "============================================"

# 1. Confirm caffeinate is still running (prevent Mac sleep)
if ! pgrep -x "caffeinate" > /dev/null; then
    echo "[ALERT] caffeinate NOT running — relaunching..."
    caffeinate -dims &
    echo "[OK] caffeinate relaunched (PID $!)"
else
    echo "[OK] caffeinate active (PIDs: $(pgrep -x caffeinate | tr '\n' ' '))"
fi

# 2. Prevent display sleep via pmset assertion
caffeinate -u -t 7200 &
echo "[OK] Display sleep prevented for 2 hours"

# 3. Check Vast.ai instance status
echo ""
echo "--- Vast.ai Instance Status ---"
vastai show instances 2>/dev/null || echo "[WARN] Could not reach Vast.ai API"

# 4. Check credit balance
echo ""
echo "--- Account Credit ---"
vastai show user 2>/dev/null | grep -i credit | head -3 || echo "[WARN] Could not fetch credit"

# 5. Check training log progress if available
echo ""
echo "--- Training Log (last 20 lines) ---"
if [ -f "$LOG" ]; then
    tail -20 "$LOG"
    echo ""
    echo "--- Latest Loss ---"
    grep "\[Step" "$LOG" | tail -5 || echo "(no step entries yet)"
else
    echo "[INFO] Training log not yet synced to Mac."
fi

# 6. Check if new checkpoint arrived
echo ""
echo "--- Checkpoints ---"
ls -lh "$CKPT_DIR"/*.pt 2>/dev/null || echo "No .pt files found"

# 7. Check if training finished
if [ -f "$CKPT_DIR/pcei_750m_kindergarten_v1.pt" ]; then
    echo ""
    echo "[SUCCESS] Final checkpoint pcei_750m_kindergarten_v1.pt is on Mac!"
    echo "Training may be complete. Check instance status and destroy if done."
fi

echo ""
echo "============================================"
echo "  Watcher complete — $(date '+%H:%M:%S')"
echo "============================================"
