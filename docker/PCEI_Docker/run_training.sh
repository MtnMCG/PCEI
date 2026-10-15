#!/bin/bash
set -e

# Setup Data
echo "============================================================"
echo "  [1/4] INITIALIZING MASSIVE DATA ACQUISITION (1B Tokens)  "
echo "============================================================"
python3 /app/setup_data.py

echo "============================================================"
echo "  [2/4] PHASE 1: BOOTSTRAP (150,000 Steps)                 "
echo "============================================================"
cd /app/pcei_framework
python3 scripts/train.py \
    --checkpoint-name pcei_500m.pt \
    --bootstrap-steps 150000 \
    --integration-steps 0 \
    --open-steps 0 \
    --high-quality-only \
    --batch-size 8 \
    --grad-accum-steps 4 \
    --lr 5e-4

echo "============================================================"
echo "  [3/4] PHASE 2: INTEGRATION (80,000 Steps)                "
echo "============================================================"
python3 scripts/train.py \
    --checkpoint-name pcei_500m.pt \
    --bootstrap-steps 0 \
    --integration-steps 80000 \
    --open-steps 0 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --lr 1e-4

echo "============================================================"
echo "  [4/4] PHASE 3: OPEN MODE (20,000 Steps)                  "
echo "============================================================"
python3 scripts/train.py \
    --checkpoint-name pcei_500m.pt \
    --bootstrap-steps 0 \
    --integration-steps 0 \
    --open-steps 20000 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --lr 5e-5

echo "============================================================"
echo "  TRAINING COMPLETE! PCEI FLIGHT READY.                    "
echo "============================================================"
