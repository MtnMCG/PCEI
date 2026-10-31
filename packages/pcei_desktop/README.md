# PCEI 750M Desktop Edition

[![License: MIT-NC](https://img.shields.io/badge/License-MIT--NC-blue.svg)](https://github.com/MtnMCG/PCEI/blob/main/LICENSE)
[![Architecture: Selective SSM + BitNet 1.58b](https://img.shields.io/badge/Architecture-Mamba%20SSM%20%2B%20BitNet%201.58b-red.svg)]()
[![Complexity: O(1) Memory](https://img.shields.io/badge/Inference-O(1)%20Memory-green.svg)]()

A standalone, cross-platform distribution of the **Physically Constrained Emergent Intelligence (PCEI) 750M** neural architecture (`886M` physical parameters: 24-layer Mamba Selective State Space + FastKAN Spline Manifolds + BitNet 1.58b Ternary Quantization).

---

## Quick Start (One-Click Launch)

### macOS
1. Double-click **`run_mac.command`** in Finder (or execute `./run_mac.command` in Terminal).
2. It will automatically initialize a clean Python environment, verify the model weights, and open the interactive terminal chat.

### Windows
1. Double-click **`run_windows.bat`**.
2. It will open Command Prompt, initialize the virtual environment, and launch the chat.

### Linux
1. Run **`./run_linux.sh`** (or double-click if your desktop environment supports executing shell scripts).
2. It will automatically detect your desktop terminal emulator (GNOME Terminal, Konsole, Alacritty, xterm) and start the session.

---

## Features
- **Strict $\mathcal{O}(1)$ Memory Recurrence**: 24-layer Mamba selective state spaces maintain constant inference memory regardless of session length.
- **BitNet 1.58b Ternary Weights**: $\mathbf{W}_q \in \{-1, 0, +1\}$ matrix operations across 332M ternary synapses.
- **Multi-Turn Dialogue Memory Active**: Sliding-window context maintains dialogue coherence across multi-turn exchanges.
- **Default Sampling**: Temperature `0.5`, Top-p `0.9`, Repetition Penalty `1.05`.
- **Auto-Download Fallback**: If `checkpoints/pcei_750m_competitive_best.pt` is not present, `chat.py` automatically downloads the weights from Hugging Face (`MtnMCG/PCEI-Text`).

---

## Interactive In-Chat Commands
- `/clear` or `/reset` — Clear the active conversation history.
- `/history` — Display all stored conversation turns in the active context.
- `/help` — Display command help.
- `/exit` or `quit` — Close the session.

---

## Manual Execution
If you prefer running directly from your own Python environment:
```bash
pip install -r requirements.txt
python3 chat.py --temperature 0.5 --top-p 0.9
```

---

## 📜 License & Intellectual Property
This distribution and the PCEI architecture are licensed under the **MIT Non-Commercial License (MIT-NC)**.
- **Academic, Educational & Personal Research**: Freely granted without fee or restriction.
- **Commercial Use & Monetization**: Strictly prohibited without prior express written permission from the copyright holder.
