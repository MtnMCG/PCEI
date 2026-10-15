#!/usr/bin/env python3
"""
PCEI 750M — Live Cloud Training Terminal Monitor
Standalone dashboard running in native macOS Terminal.
Reads locally synced telemetry from RTX 4090 in real-time.
"""

import os
import sys
import time
import glob
from datetime import datetime
from pathlib import Path

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_DIM = "\033[2m"

LOGS_BASE = str(Path(__file__).resolve().parent / "src" / "logs")

def get_latest_csv():
    csv_files = glob.glob(f"{LOGS_BASE}/*/loss_summary.csv")
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)

def render():
    os.system("clear")
    csv_path = get_latest_csv()
    
    print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     🧠 PCEI 750M — AUTONOMOUS RTX 4090 CLOUD TRAINING MONITOR            ║{C_RESET}")
    host_display = os.environ.get("REMOTE_HOST", "Remote Cloud GPU")
    print(f"{C_DIM} Hardware: NVIDIA GeForce RTX 4090 (24GB) | Host: {host_display}{C_RESET}")
    print(f"{C_DIM} Local Time: {datetime.now().strftime('%A, %b %d %I:%M:%S %p')} | Refresh: 2s{C_RESET}")
    print(f"{C_CYAN}────────────────────────────────────────────────────────────────────────────{C_RESET}")

    if not csv_path or not os.path.exists(csv_path):
        print(f"\n {C_YELLOW}Waiting for initial telemetry sync from cloud...{C_RESET}\n")
        return

    try:
        with open(csv_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if len(lines) <= 1:
        print(f"\n {C_YELLOW}Awaiting first loss step records...{C_RESET}\n")
        return

    rows = lines[1:]
    try:
        total_steps = int(rows[-1].split(",")[0])
    except (ValueError, IndexError):
        total_steps = len(rows)

    init_loss = float(rows[0].split(",")[2])
    recent_losses = [float(r.split(",")[2]) for r in rows[-20:]]
    latest_loss = recent_losses[-1]
    avg_20 = sum(recent_losses) / len(recent_losses)
    min_loss = min(float(r.split(",")[2]) for r in rows)
    loss_reduction = init_loss - latest_loss

    # Progress bar targeting 250k Phase 1 milestone
    target_steps = 250000
    pct = min(100.0, (total_steps / target_steps) * 100)
    bar_width = 30
    filled = int((pct / 100.0) * bar_width)
    bar = f"{C_GREEN}{'█' * filled}{C_DIM}{'░' * (bar_width - filled)}{C_RESET}"

    print(f" {C_BOLD}Global Step:{C_RESET} {C_GREEN}{total_steps:,}{C_RESET} / {target_steps:,} ({pct:.1f}%)   [{bar}]")
    print()
    print(f" {C_BOLD}Loss Metrics:{C_RESET}")
    print(f"   • Starting Loss:      {C_YELLOW}{init_loss:.4f}{C_RESET}")
    print(f"   • Latest Step Loss:   {C_BOLD}{C_GREEN}{latest_loss:.4f}{C_RESET}")
    print(f"   • 20-Step Moving Avg: {C_CYAN}{avg_20:.4f}{C_RESET}")
    print(f"   • Best Loss Recorded: {C_MAGENTA}{min_loss:.4f}{C_RESET}")
    print(f"   • Total Drop:         {C_GREEN}-{loss_reduction:.4f} pts{C_RESET}")
    print()
    print(f" {C_BOLD}Throughput & Economics:{C_RESET}")
    print(f"   • Step Speed:         ~0.50s / step (RTX 4090 Tensor Cores)")
    print(f"   • Processing Rate:    ~2,048 tokens / sec  (~7.37M tokens / hr)")
    print(f"   • Compute Burn:       $0.353 / hour")
    print(f"   • Telemetry Sync:     {C_GREEN}Active (every 60s to MacBook){C_RESET}")
    print()
    print(f"{C_CYAN}────────────────────────────────────────────────────────────────────────────{C_RESET}")
    print(f" {C_BOLD}Recent Step Trajectory (Last 10 updates):{C_RESET}")
    recent_tail = rows[-10:]
    for r in recent_tail:
        parts = r.split(",")
        step_idx = parts[0]
        loss_val = float(parts[2])
        # Simple visual spark
        spark = " " if loss_val > 9 else ("▃" if loss_val > 8 else ("▄" if loss_val > 7 else "▆"))
        print(f"   Step {int(step_idx):5d} | Loss: {loss_val:.4f} {C_GREEN}{spark}{C_RESET}")

    print(f"{C_CYAN}────────────────────────────────────────────────────────────────────────────{C_RESET}")
    print(f"{C_DIM}Press Ctrl+C anytime to close this monitor (training will keep running).{C_RESET}")

def main():
    while True:
        try:
            render()
            time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Monitor detached. Background training continues unaffected.{C_RESET}\n")
            break

if __name__ == "__main__":
    main()
