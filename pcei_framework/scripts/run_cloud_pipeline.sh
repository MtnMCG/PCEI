#!/usr/bin/env bash
set -e
cd /workspace/pcei

echo "================================================================"
echo "  PCEI TOKEN-MAXX INSTRUCTION & REASONING PIPELINE"
echo "  Foundation Teacher Anchor: ACTIVE (35% weight)"
echo "  Date: $(date)"
echo "================================================================"

# Base Checkpoint: pcei_750m_reasoning_step_120000.pt
echo "[Pipeline $(date)] Resuming from step 120,000 checkpoint..."

# 1. TOKEN-MAXX BLENDED INSTRUCTION TUNING TILL $2.00 CREDIT LIMIT
echo "[Pipeline $(date)] Launching Token-Maxx Blended Instruction & Reasoning Tuner..."
python3 tools/blended_instruction_tuner.py \
    --checkpoint-name pcei_750m_reasoning_step_120000.pt \
    --output-name pcei_750m_reasoning_assistant.pt \
    --batch-size 2 \
    --grad-accum-steps 2 \
    --save-every 10000 \
    --probe-every 2500 \
    --eval-every 1000 \
    --steps 150000 \
    --base-lr 6.0e-5 \
    --min-lr 1.0e-5 \
    --device cuda 2>&1 | tee /workspace/blended_instruction.log

echo "[Pipeline $(date)] Instruction Tuning Complete or Credit Floor Reached!"

# 2. COMPREHENSIVE FINAL EVALUATION & BENCHMARKS
echo "[Pipeline $(date)] Running Extensive Generation Test Suite..."
python3 scripts/test_extensive_generation.py \
    --ckpt checkpoints/pcei_750m_reasoning_assistant.pt \
    --device cuda 2>&1 | tee /workspace/final_generation_eval.log

echo "[Pipeline $(date)] Running HellaSwag Evaluation..."
python3 scripts/eval_hellaswag.py \
    --ckpt checkpoints/pcei_750m_reasoning_assistant.pt \
    --device cuda 2>&1 | tee /workspace/final_hellaswag_eval.log

echo "PIPELINE_COMPLETE=$(date)" > /workspace/pipeline_all_finished.txt
echo "[Pipeline $(date)] Entire Training, Alignment, and Evaluation Pipeline Finished Successfully!"
