#!/usr/bin/env bash
set -e
cd /workspace/pcei

echo "================================================================"
echo "  PCEI CONVERSATIONAL REASONING ALIGNMENT PIPELINE"
echo "  Date: $(date)"
echo "  Base Checkpoint: pcei_750m_sanitized_base.pt"
echo "================================================================"

# 1. Install prerequisites
echo "[Pipeline $(date)] Installing runtime packages..."
pip install --no-cache-dir tiktoken einops

# 2. Launch Credit Guard Daemon in Background
echo "[Pipeline $(date)] Launching Credit Guard Daemon..."
python3 scripts/credit_guard_daemon.py > /workspace/credit_guard.log 2>&1 &
GUARD_PID=$!
echo "[Pipeline] Credit Guard running with PID $GUARD_PID"

# 3. Launch Blended Instruction Tuner (Conversational Reasoning Mode)
echo "[Pipeline $(date)] Launching 2,500-step Conversational Reasoning Alignment..."
python3 tools/blended_instruction_tuner.py     --checkpoint-name pcei_750m_sanitized_base.pt     --output-name pcei_750m_aligned_v2.pt     --curriculum-mode conversational_reasoning     --batch-size 2     --grad-accum-steps 4     --max-seq-len 2048     --steps 2500     --base-lr 2.0e-5     --min-lr 5.0e-6     --eval-every 250     --probe-every 500     --save-every 1000     --device cuda 2>&1 | tee /workspace/alignment_training.log

echo "[Pipeline $(date)] Alignment training loop completed!"

# 4. Run Probe / Verification Suite if script exists
if [ -f scripts/test_extensive_generation.py ]; then
    echo "[Pipeline $(date)] Running Verification Probe on Aligned Model..."
    python3 scripts/test_extensive_generation.py         --ckpt checkpoints/pcei_750m_aligned_v2.pt         --device cuda 2>&1 | tee /workspace/final_aligned_probe.log || true
fi

echo "ALIGNMENT_FINISHED=$(date)" > /workspace/alignment_finished.txt
echo "[Pipeline $(date)] Pipeline fully finished! Ready for weight sync."
