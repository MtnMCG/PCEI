#!/bin/bash
# Prolonged 30,000 step training run for the validated 119M PCEI model

set -e

cd "$(dirname "$0")/.."

echo "Starting 30,000 step training run on the validated 119M architecture..."

./venv/bin/python scripts/train.py \
  --bootstrap-steps 10000 \
  --integration-steps 10000 \
  --open-steps 10000 \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --diffusion-steps 3 \
  --d-model-text 512 \
  --d-model-img 512 \
  --d-model-audio 256 \
  --layers-text 8 \
  --layers-img 4 \
  --layers-audio 2 \
  --log-every 50 \
  --eval-every 500 \
  --plot-every 1000 \
  --checkpoint-name pcei_checkpoint_119m_30k_run.pt
