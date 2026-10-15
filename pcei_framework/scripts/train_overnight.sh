#!/bin/bash
# Overnight target: ~6350 steps to finish by 7:00 AM local time.
# Model scale: 119M Parameters
# Estimated duration: ~12 hours @ 6.8 seconds/step

set -e

cd "$(dirname "$0")/.."

echo "Starting targeted overnight 119M training run..."
echo "Target completion: 7:00 AM (approx. 6350 steps at 6.8s/step)"

./venv/bin/python scripts/train.py \
  --bootstrap-steps 2000 \
  --integration-steps 2000 \
  --open-steps 2350 \
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
  --eval-every 200 \
  --plot-every 500 \
  --checkpoint-name pcei_checkpoint_overnight_119m.pt
