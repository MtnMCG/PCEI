#!/bin/bash
# PCEI "Unleashed" Training Session
# 300 Steps: 150 Bootstrap + 150 Integration
# Learnable Ternary Weights + Physical Topology Migration Active

echo "🚀 Unleashing the PCEI Beast..."
echo "Config: 150 Bootstrap Steps | 150 Integration Steps"
echo "Features: Learnable Sparsity + Topological Drift Active"

# Activate the virtual environment
source venv/bin/activate

# Use caffeinate to prevent sleep during training
caffeinate -i -s ./venv/bin/python scripts/train.py \
    --lr 3e-4 \
    --batch-size 4 \
    --bootstrap-steps 150 \
    --integration-steps 150 \
    --open-steps 0 \
    --log-every 10 \
    --eval-every 25 \
    --plot-every 50 \
    --resume

echo "✅ Unleashed session concluded."
