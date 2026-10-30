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
    echo "[*] Installing required runtime dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[*] Launching PCEI Resonance Chat..."
python3 chat.py "$@"
