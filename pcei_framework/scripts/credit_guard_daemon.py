#!/usr/bin/env python3
import time
import urllib.request
import json
import os
import signal
import subprocess

API_KEY = os.environ.get("VAST_API_KEY", "")
THRESHOLD = 1.50

def prune_step_logs():
    """Removes bloated per-step snapshot json files, preserving loss_summary.csv, pngs, and canonical jsons."""
    try:
        subprocess.run(["find", "/workspace/pcei/src/logs", "-name", "*_step*.json", "-delete"], check=False)
    except Exception as e:
        print(f"[CreditGuard] Notice: Log prune exception: {e}")

def get_credit():
    if not API_KEY:
        print("[CreditGuard] Warning: VAST_API_KEY environment variable is not set.")
        return None
    try:
        req = urllib.request.Request(
            "https://console.vast.ai/api/v0/users/current/",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("credit", 0.0))
    except Exception as e:
        print(f"[CreditGuard] Warning: Failed to query credit: {e}")
        return None

def find_train_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "blended_instruction_tuner.py"]).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        if pids:
            return pids[0]
    except Exception:
        pass
    try:
        out = subprocess.check_output(["pgrep", "-f", "scripts/train.py"]).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None

def main():
    print(f"[CreditGuard] Autonomous Credit & Disk Guard active (Threshold: ${THRESHOLD:.2f}).")
    while True:
        prune_step_logs()
        bal = get_credit()
        if bal is not None:
            print(f"[CreditGuard] Current balance: ${bal:.2f}")
            if bal <= THRESHOLD:
                print(f"[CreditGuard] ALERT: Credit ${bal:.2f} <= ${THRESHOLD:.2f} threshold! Safe shutdown...")
                with open("/workspace/pcei/LOW_CREDIT_STOP", "w") as f:
                    f.write(f"LOW_CREDIT_STOP: {bal}\n")
                pid = find_train_pid()
                if pid:
                    print(f"[CreditGuard] Sending SIGINT to train.py (PID {pid}) to trigger calcified checkpoint save...")
                    os.kill(pid, signal.SIGINT)
                break
        time.sleep(600)

if __name__ == "__main__":
    main()
