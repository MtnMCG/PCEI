#!/bin/bash
# High-scale training config for PCEI
# This runs the model at roughly a 500M parameter scale for an extended duration.

set -e

cd "$(dirname "$0")/.."

echo "Starting massive 500M scale prolonged training run..."

./venv/bin/python scripts/train.py \
  --bootstrap-steps 2000 \
  --integration-steps 4000 \
  --open-steps 4000 \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --diffusion-steps 5 \
  --d-model-text 1024 \
  --d-model-img 512 \
  --d-model-audio 512 \
  --layers-text 16 \
  --layers-img 8 \
  --layers-audio 4 \
  --log-every 50 \
  --eval-every 200 \
  --plot-every 500 \
  --checkpoint-name pcei_checkpoint_massive_500m.pt
