#!/usr/bin/env bash
# ==============================================================================
# PCEI 750M Desktop Launcher for Linux
# Auto-detects terminal emulator and launches PCEI Resonance Chat.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

# If not running in a terminal, spawn one
if [ ! -t 0 ]; then
    for TERM_EMU in x-terminal-emulator gnome-terminal konsole xfce4-terminal kitty alacritty xterm; do
        if command -v "$TERM_EMU" &> /dev/null; then
            exec "$TERM_EMU" -e "$0" "$@"
        fi
    done
fi

echo "================================================================"
echo "  PCEI 750M Desktop — Linux Auto-Boot Terminal Launcher"
echo "================================================================"

# Verify Python 3
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed."
    echo "    Please install it via: sudo apt install python3 python3-venv"
    read -p "Press Enter to exit..."
    exit 1
fi

# Set up local virtual environment if not present
if [ ! -d "venv" ]; then
    echo "[*] Initializing local environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    if command -v nvidia-smi &> /dev/null; then
        echo "[*] NVIDIA GPU detected! Installing CUDA-accelerated PyTorch..."
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    else
        pip install torch
    fi
    pip install tiktoken huggingface_hub
else
    source venv/bin/activate
    if command -v nvidia-smi &> /dev/null; then
        if ! python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "[*] NVIDIA GPU detected! Upgrading to CUDA-accelerated PyTorch..."
            pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
        fi
    fi
fi

echo "[*] Launching PCEI Resonance Chat (GPU Auto-Accelerated, Temp=0.5, Multi-Turn History)..."
python3 chat.py --temperature 0.5 --top-p 0.9 --repetition-penalty 1.05 "$@"
