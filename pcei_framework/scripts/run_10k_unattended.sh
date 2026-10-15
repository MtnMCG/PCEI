#!/bin/bash
# script: run_10k_unattended.sh
# Purpose: Executes the PCEI 7,000-step training run and prevents macOS from going to sleep.

echo "=========================================================="
echo " Starting 7,000 Step PCEI Training Run"
echo " macOS Sleep Prevention: ACTIVE (caffeinate)"
echo "=========================================================="
echo ""

# Navigate to project root to ensure standard execution
cd "$(dirname "$0")/.."

# Activate the virtual environment
source venv/bin/activate

# Use caffeinate to prevent:
# -i : idle sleep (keeps CPU awake even if inactive)
# -s : system sleep (keeps system awake when plugged in)
# The training command will run inside this wrapper.
caffeinate -i -s python scripts/train.py \
    --lr 3e-4 \
    --batch-size 4 \
    --bootstrap-steps 0 \
    --integration-steps 3500 \
    --open-steps 0 \
    --resume

echo ""
echo "=========================================================="
echo " Training Complete!"
echo " Sleep prevention deactivated. System can now rest."
echo "=========================================================="
