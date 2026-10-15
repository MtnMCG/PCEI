#!/usr/bin/env bash
set -e
cd /workspace/pcei

echo "================================================================"
echo "  PCEI TARGETED REASONING ANNEALING PIPELINE"
echo "  Date: $(date)"
echo "  Base Checkpoint: pcei_750m_aligned_v2.pt"
echo "  Target: 1,000 steps on 50% Reasoning Exemplars + 50% Claude-CoT"
echo "================================================================"

python3 tools/blended_instruction_tuner.py \
    --checkpoint-name pcei_750m_aligned_v2.pt \
    --output-name pcei_750m_reasoning_master.pt \
    --curriculum-mode targeted_annealing \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --max-seq-len 2048 \
    --steps 1000 \
    --base-lr 1.0e-5 \
    --min-lr 2.0e-6 \
    --eval-every 100 \
    --probe-every 250 \
    --save-every 500 \
    --device cuda 2>&1 | tee /workspace/annealing_training.log

echo "ANNEALING_FINISHED=$(date)" > /workspace/annealing_finished.txt
echo "[Pipeline $(date)] Targeted Annealing fully finished! Master model ready."
