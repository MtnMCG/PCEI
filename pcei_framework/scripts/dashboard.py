#!/usr/bin/env python3
"""
scripts/dashboard.py
PCEI Real-Time Terminal HUD Dashboard
Fetches live telemetry from the remote RTX 4090 and local system every 3 seconds.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

SSH_KEY = Path(os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")))
REMOTE_HOST = os.environ.get("REMOTE_HOST", "<REMOTE_GPU_IP>")
REMOTE_PORT = os.environ.get("REMOTE_PORT", "22")

def fetch_telemetry():
    remote_script = (
        "python3 -c \""
        "import subprocess, json, os\n"
        "from pathlib import Path\n"
        "res = {}\n"
        "# 1. GPU info\n"
        "try:\n"
        "    s = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], text=True).strip()\n"
        "    u, mu, mt, temp, pwr = [x.strip() for x in s.split(',')]\n"
        "    res['gpu'] = {'util': u + '%', 'vram': f'{int(mu)/1024:.1f}/{int(mt)/1024:.1f} GB', 'temp': temp + 'C', 'pwr': pwr + 'W'}\n"
        "except Exception as e:\n"
        "    res['gpu'] = {'error': str(e)}\n"
        "# 2. Credit\n"
        "res['credit'] = 'N/A'\n"
        "cp = Path('/workspace/credit_guard.log')\n"
        "if cp.exists():\n"
        "    for l in reversed(cp.read_text().splitlines()):\n"
        "        if 'Current balance: $' in l:\n"
        "            res['credit'] = l.split('Current balance: $')[-1].strip()\n"
        "            break\n"
        "# 3. Recent training steps\n"
        "lp = Path('/workspace/blended_instruction.log')\n"
        "res['recent_steps'] = []\n"
        "if lp.exists():\n"
        "    lines = lp.read_text().splitlines()[-30:]\n"
        "    for l in lines:\n"
        "        if '[Step ' in l:\n"
        "            res['recent_steps'].append(l)\n"
        "# 4. Teacher samples synced\n"
        "tp = Path('/workspace/pcei/data/synthetic/pcei_foundation_teacher.jsonl')\n"
        "res['teacher_samples'] = sum(1 for _ in open(tp)) if tp.exists() else 0\n"
        "print(json.dumps(res))\n"
        "\""
    )

    cmd = [
        "ssh", "-i", str(SSH_KEY), "-p", REMOTE_PORT,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
        f"root@{REMOTE_HOST}", remote_script
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        import json
        for line in out.splitlines():
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
    except Exception:
        pass
    return None

def main():
    try:
        while True:
            data = fetch_telemetry()
            os.system("clear")
            print("=" * 76)
            print("  ⚡ PCEI REAL-TIME TRAINING HUD — RTX 4090 CLOUD INSTANCE ⚡")
            print("=" * 76)

            if not data:
                print(f"\n  [Connecting to remote GPU instance {REMOTE_HOST}...]")
            else:
                gpu = data.get("gpu", {})
                credit = data.get("credit", "N/A")
                teacher = data.get("teacher_samples", 0)

                print(f"  GPU Utilization : {gpu.get('util', 'N/A'):<10} | VRAM : {gpu.get('vram', 'N/A'):<15} | Temp : {gpu.get('temp', 'N/A')}")
                print(f"  Power Draw      : {gpu.get('pwr', 'N/A'):<10} | Cloud Credit : ${credit:<10} (Stop Floor: $2.00)")
                print(f"  Teacher Curriculum : {teacher} high-density samples synced on remote host")
                print("-" * 76)
                print("  LATEST TRAINING TELEMETRY (Loss, Perplexity, Tok/s, Active Bonds):")
                print("-" * 76)
                steps = data.get("recent_steps", [])
                if steps:
                    for s in steps[-6:]:
                        print(f"  {s}")
                else:
                    print("  Awaiting next telemetry step...")
            print("=" * 76)
            print("  [Auto-refreshing every 3s | Press Ctrl+C to exit dashboard]")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n[Dashboard] Exited. Cloud training continues in background.")

if __name__ == "__main__":
    main()
