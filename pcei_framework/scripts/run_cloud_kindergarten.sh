#!/usr/bin/env bash
# scripts/run_cloud_kindergarten.sh
# Cloud kindergarten training pipeline.
# - Trains from pcei_750m_aligned_v2.pt
# - Saves to pcei_750m_kindergarten_v1.pt (NEVER touches v2)
# - Syncs checkpoints every 500 steps back to Mac via rsync

set -e
cd /workspace/pcei

echo "================================================================"
echo "  PCEI KINDERGARTEN ALIGNMENT — TinyStories + Arithmetic"
echo "  Date: $(date)"
echo "  Base: pcei_750m_aligned_v2.pt -> Out: pcei_750m_kindergarten_v1.pt"
echo "================================================================"

# Install prerequisites
echo "[Pipeline] Installing runtime packages..."
pip install --no-cache-dir tiktoken einops 2>&1 | tail -5

# Budget guard: kill if credit falls below $0.20 (belt + suspenders on top of credit_guard)
python3 -c "
import subprocess, time, sys
while True:
    try:
        out = subprocess.check_output(['vastai','show','user'], text=True)
        for line in out.splitlines():
            if 'credit' in line.lower():
                import re
                m = re.findall(r'[0-9]+\.[0-9]+', line)
                if m:
                    credit = float(m[0])
                    if credit < 0.20:
                        print(f'[CreditGuard] CREDIT LOW ({credit:.2f}) — triggering shutdown.')
                        subprocess.run(['pkill','-f','blended_instruction_tuner'], check=False)
                        sys.exit(0)
    except Exception:
        pass
    time.sleep(120)
" > /workspace/credit_guard.log 2>&1 &
GUARD_PID=$!
echo "[Pipeline] Credit guard running (PID $GUARD_PID)"

# MAC_IP auto-detected from SSH_CLIENT
MAC_IP=$(echo $SSH_CLIENT | awk '{print $1}')
echo "[Sync] Mac IP detected: $MAC_IP"
MAC_CKPT_DIR="/Users/emerson/Desktop/PCEI /pcei_framework/checkpoints"
MAC_LOG_DIR="/Users/emerson/Desktop/PCEI /pcei_framework"

# Background sync daemon — runs every 10 minutes, copies kindergarten checkpoints only
(while true; do
    sleep 600
    echo "[Sync $(date)] Syncing kindergarten checkpoints to Mac..."
    rsync -avz --progress \
        --include="pcei_750m_kindergarten_*.pt" \
        --include="cloud_kindergarten_training.log" \
        --exclude="*" \
        /workspace/pcei/checkpoints/ \
        /workspace/pcei/cloud_kindergarten_training.log \
        root@${MAC_IP}:"${MAC_CKPT_DIR}/" 2>&1 | tail -5 || echo "[Sync] rsync failed, will retry."
done) > /workspace/sync_daemon.log 2>&1 &
SYNC_PID=$!
echo "[Sync Daemon] Running (PID $SYNC_PID)"

# Compute max steps from budget: ~$2.40 usable / $0.3996/hr * 3600s / ~14s per step ≈ 1542 steps
# Being conservative: use 1200 steps to leave buffer for storage + bandwidth
# Each step = 1 grad accum cycle = 4096 effective tokens
MAX_STEPS=1200

echo "[Pipeline] Launching kindergarten training for $MAX_STEPS steps..."
python3 tools/blended_instruction_tuner.py \
    --checkpoint-name pcei_750m_aligned_v2.pt \
    --output-name pcei_750m_kindergarten_v1.pt \
    --curriculum-mode kindergarten \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --max-seq-len 1024 \
    --steps $MAX_STEPS \
    --base-lr 1.5e-5 \
    --min-lr 3.0e-6 \
    --eval-every 100 \
    --probe-every 200 \
    --save-every 400 \
    --device cuda 2>&1 | tee /workspace/pcei/cloud_kindergarten_training.log

echo "[Pipeline $(date)] Training complete!"

# Final sync of finished checkpoint
echo "[Sync] Final sync to Mac..."
rsync -avz --progress \
    /workspace/pcei/checkpoints/pcei_750m_kindergarten_v1.pt \
    /workspace/pcei/cloud_kindergarten_training.log \
    root@${MAC_IP}:"${MAC_CKPT_DIR}/" 2>&1

echo "KINDERGARTEN_FINISHED=$(date)" > /workspace/kindergarten_finished.txt
echo "[Pipeline] All done. Instance can now be destroyed."
