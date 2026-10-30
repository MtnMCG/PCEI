#!/usr/bin/env bash
# ==============================================================================
# PCEI 750M Desktop Launcher for macOS
# Double-click this file in Finder to launch PCEI Resonance Chat in Terminal.
# ==============================================================================

cd "$(dirname "$0")" || exit 1

echo "================================================================"
echo "  PCEI 750M Desktop — macOS Auto-Boot Terminal Launcher"
echo "================================================================"

# Verify Python 3
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: Python 3 was not found on your system."
    echo "    Please install Python 3 via https://www.python.org or 'brew install python'"
    read -p "Press Enter to exit..."
    exit 1
fi

# Set up local virtual environment if not present
if [ ! -d "venv" ]; then
    echo "[*] Initializing local environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[*] Installing required runtime dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set MPS memory optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "[*] Launching PCEI Resonance Chat..."
python3 chat.py "$@"
