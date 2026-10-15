#!/usr/bin/env bash
cd /workspace/pcei

echo "================================================================="
echo " Starting PCEI 750M 512-Context Calibrated Supervisor on RTX 4090"
echo " Context: 512 Tokens | Batch: 2 | Accum: 8 (Effective: 8,192 tok/step)"
echo " Calibrated Schedule: 250,000 Boot | 15,000 Integ | 50,000 Dream"
echo " Started at: $(date)"
echo "================================================================="

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if ! pgrep -f "credit_guard_daemon.py" > /dev/null; then
    nohup python3 -u /workspace/pcei/credit_guard_daemon.py > /workspace/pcei/credit_guard.log 2>&1 &
fi

while true; do
    if [ -f "/workspace/pcei/LOW_CREDIT_STOP" ]; then
        echo "[BUDGET GUARD] Low credit stop flag detected. Halting supervisor gracefully."
        break
    fi

    echo "[$(date)] Launching / Resuming PCEI Training (512 Context, Accum 8, 250k Boot, 15k Integ, 50k Dream)..."
    python3 scripts/train.py \
        --checkpoint-name pcei_750m_supercharged.pt \
        --max-seq-len 512 \
        --batch-size 2 \
        --diffusion-steps 1 \
        --bootstrap-steps 250000 \
        --integration-steps 15000 \
        --open-steps 50000 \
        --pack-sequences \
        --schedule wsd \
        --lr 3e-4 \
        --grad-accum-steps 8 \
        --log-every 25 \
        --eval-every 250 \
        --plot-every 250 \
        2>&1 | tee -a /workspace/train.log
    
    EXIT_CODE=$?
    echo "[$(date)] Process exited with code $EXIT_CODE."

    if [ $EXIT_CODE -eq 0 ] && [ ! -f "/workspace/pcei/LOW_CREDIT_STOP" ]; then
        echo "[$(date)] 🎉 All 3 PCEI training phases completed successfully!"
        cp /workspace/pcei/checkpoints/pcei_750m_supercharged.pt /workspace/pcei/checkpoints/pcei_750m_final_completed.pt
        break
    fi

    echo "Clearing VRAM and auto-restarting in 5s..."
    sleep 5
done
