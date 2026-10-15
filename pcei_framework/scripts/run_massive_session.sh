#!/usr/bin/env bash
set -e
cd /workspace/pcei

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/workspace/pcei:$PYTHONPATH"

echo "================================================================="
echo " Starting PCEI 750M Massive Autonomous Multi-Phase Training Run"
echo " Remaining Vast.ai Budget: ~$15.00 (Runway: ~35-38 Hours)"
echo " Resuming from Checkpoint: pcei_750m_3phase_completed.pt (Step 91,251)"
echo " Total Plan: 150,000 Steps across 3 Phases (50k each)"
echo " Started at: $(date)"
echo "================================================================="

# 1. Ensure Credit Guard is actively guarding the balance
if ! pgrep -f "credit_guard_daemon.py" > /dev/null; then
    echo "[CreditGuard] Launching Autonomous Credit & Disk Guard..."
    nohup python3 -u /workspace/pcei/credit_guard_daemon.py > /workspace/credit_guard.log 2>&1 &
fi

# 2. Sync checkpoint if needed
if [ ! -f /workspace/pcei/checkpoints/pcei_750m_supercharged.pt ] && [ -f /workspace/pcei/checkpoints/pcei_750m_3phase_completed.pt ]; then
    cp /workspace/pcei/checkpoints/pcei_750m_3phase_completed.pt /workspace/pcei/checkpoints/pcei_750m_supercharged.pt
fi

# 3. Launch Massive Training Session (50k steps / phase = 150k steps total)
# Phase 1: WitnessMode Text Healing (91,251 -> 141,251) on 541,239 Claude/Council/Synthetic texts
# Phase 2: Clean Multimodal Integration (141,251 -> 191,251) on 6,917 verified COCO captions & LibriSpeech
# Phase 3: Deep Consolidation & Dreaming (191,251 -> 241,251) on AmbientField slow state
python3 scripts/train.py \
    --checkpoint-name pcei_750m_supercharged.pt \
    --steps-per-phase 50000 \
    --max-seq-len 512 \
    --batch-size 2 \
    --diffusion-steps 1 \
    --pack-sequences \
    --schedule wsd \
    --lr 2.5e-4 \
    --grad-accum-steps 8 \
    --log-every 25 \
    --eval-every 500 \
    --plot-every 500 \
    2>&1 | tee -a /workspace/train_massive.log

echo "[Massive Run] Completed training successfully at: $(date)!"
cp /workspace/pcei/checkpoints/pcei_750m_supercharged.pt /workspace/pcei/checkpoints/pcei_750m_massive_session_final.pt

# 4. Final Comprehensive Benchmark Suite
echo "================================================================="
echo " Executing Final Multimodal & Language Benchmark Suite"
echo "================================================================="
python3 scripts/test_multimodal_and_benchmarks.py pcei_750m_supercharged.pt 2>&1 | tee /workspace/benchmark_massive_final.log
