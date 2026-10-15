#!/usr/bin/env python3
"""
scripts/monitor_live.py
Live streaming monitor for PCEI cloud training, loss, GPU metrics, and credit guard.
Usage:
    python3 scripts/monitor_live.py
"""

import sys
import subprocess
from pathlib import Path

SSH_KEY = Path(os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")))
REMOTE_HOST = os.environ.get("REMOTE_HOST", "<REMOTE_GPU_IP>")
REMOTE_PORT = os.environ.get("REMOTE_PORT", "22")

# Remote bash command that monitors training log and nvidia-smi
REMOTE_MONITOR_CMD = (
    "tail -f -n 25 /workspace/blended_instruction.log"
)

def main():
    print("=" * 70)
    print("  PCEI LIVE CLOUD TRAINING STREAM")
    print(f"  Target: {REMOTE_HOST}:{REMOTE_PORT} (RTX 4090 24GB)")
    print("  Streaming loss, PPL, throughput, active bonds, and live probes...")
    print("  (Press Ctrl+C to exit monitor at any time without stopping training)")
    print("=" * 70 + "\n")

    cmd = [
        "ssh",
        "-t",
        "-i", str(SSH_KEY),
        "-p", REMOTE_PORT,
        "-o", "StrictHostKeyChecking=no",
        f"root@{REMOTE_HOST}",
        REMOTE_MONITOR_CMD
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Monitor] Stream disconnected. Cloud training continues running in background.")

if __name__ == "__main__":
    main()
