#!/usr/bin/env bash
# ==============================================================================
# PCEI 750M Desktop Launcher for Linux
# Auto-detects terminal emulator and launches PCEI Resonance Chat.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

if [ ! -t 0 ]; then
    for TERM_EMU in x-terminal-emulator gnome-terminal konsole xfce4-terminal kitty alacritty xterm; do
        if command -v "$TERM_EMU" &> /dev/null; then
            exec "$TERM_EMU" -e "$0" "$@"
        fi
    done
fi

echo "================================================================"
echo "  PCEI 750M Desktop - Linux Auto-Boot Terminal Launcher"
echo "================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed."
    echo "    Please install it via: sudo apt install python3 python3-venv"
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -x "venv/bin/python" ]; then
    echo "[*] Creating local environment (venv)..."
    python3 -m venv venv || { echo "[!] Failed to create venv"; read -p "Press Enter to exit..."; exit 1; }
fi
source venv/bin/activate

echo "[*] Checking dependencies and GPU support..."
python bootstrap.py || { echo "[!] Dependency setup failed. See messages above."; read -p "Press Enter to exit..."; exit 1; }

echo "[*] Launching PCEI Resonance Chat (GPU auto-detected, Temp 0.5, multi-turn history ON)..."
python chat.py --temperature 0.5 --top-p 0.9 --repetition-penalty 1.05 "$@"
