#!/usr/bin/env bash
# ==============================================================================
# PCEI 750M Desktop Launcher for macOS
# Double-click this file in Finder to launch PCEI Resonance Chat in Terminal.
# ==============================================================================

cd "$(dirname "$0")" || exit 1

echo "================================================================"
echo "  PCEI 750M Desktop - macOS Auto-Boot Terminal Launcher"
echo "================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[!] Error: Python 3 was not found on your system."
    echo "    Please install Python 3 via https://www.python.org or 'brew install python'"
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -x "venv/bin/python" ]; then
    echo "[*] Creating local environment (venv)..."
    python3 -m venv venv || { echo "[!] Failed to create venv"; read -p "Press Enter to exit..."; exit 1; }
fi
source venv/bin/activate

echo "[*] Checking dependencies..."
python bootstrap.py || { echo "[!] Dependency setup failed. See messages above."; read -p "Press Enter to exit..."; exit 1; }

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "[*] Launching PCEI Resonance Chat (Apple Silicon GPU, Temp 0.5, multi-turn history ON)..."
python chat.py --temperature 0.5 --top-p 0.9 --repetition-penalty 1.05 "$@"
