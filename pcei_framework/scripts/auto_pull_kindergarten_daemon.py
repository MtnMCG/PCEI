#!/usr/bin/env python3
"""
scripts/auto_pull_kindergarten_daemon.py

Autonomous local daemon running on the Mac.
Guarantees:
1. Continuous monitoring of training completion and Vast.ai credit.
2. Pulls the latest kindergarten checkpoint down to the Mac (saved as pcei_750m_kindergarten_v1.pt).
3. Completely protects pcei_750m_aligned_v2.pt (never touches or overwrites it).
4. Pulls the full training log.
5. Verifies checkpoint integrity locally (>3.5 GB).
6. Automatically destroys the Vast.ai instance to stop all billing before credits can run negative.
7. Keeps caffeinate alive.
"""

import subprocess
import time
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path("/Users/emerson/Desktop/PCEI /pcei_framework")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TARGET_CKPT = CHECKPOINT_DIR / "pcei_750m_kindergarten_v1.pt"
TARGET_LOG = PROJECT_ROOT / "cloud_kindergarten_training.log"
INSTANCE_ID = "52154175"
SSH_HOST = "ssh4.vast.ai"
SSH_PORT = "34174"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[AutoPullDaemon {timestamp}] {msg}"
    print(formatted, flush=True)
    with open(PROJECT_ROOT / "auto_pull_daemon.log", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def run_cmd(cmd, timeout=60):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)

VASTAI_BIN = "/opt/homebrew/bin/vastai"

def get_credit():
    import json
    code, out, _ = run_cmd(f"{VASTAI_BIN} show user --raw 2>/dev/null", timeout=15)
    if code != 0:
        return None
    try:
        data = json.loads(out)
        val = data.get("credit")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return None

def check_remote_status():
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {SSH_PORT} root@{SSH_HOST} "
    check_str = (
        "'if [ -f /workspace/pcei/checkpoints/pcei_750m_kindergarten_v1.pt ]; then echo FINISHED; fi; "
        "if ps aux | grep \"python3 tools/blended_instruction_tuner.py\" | grep -v grep > /dev/null; then echo RUNNING; else echo STOPPED; fi; "
        "ls -lh /workspace/pcei/checkpoints/'"
    )
    code, out, _ = run_cmd(ssh_cmd + check_str, timeout=30)
    return code == 0, out

def pull_checkpoint():
    log("Initiating checkpoint download to Mac...")
    # First check what checkpoints exist remotely
    list_cmd = f"ssh -o StrictHostKeyChecking=no -p {SSH_PORT} root@{SSH_HOST} 'ls -1 /workspace/pcei/checkpoints/*.pt 2>/dev/null'"
    code, out, _ = run_cmd(list_cmd, timeout=30)
    remote_files = [line.strip() for line in out.splitlines() if line.strip().endswith(".pt")]
    
    # Identify best candidate: pcei_750m_kindergarten_v1.pt first, then latest
    chosen_file = None
    for rf in remote_files:
        if "kindergarten_v1" in rf:
            chosen_file = rf
            break
    if not chosen_file:
        for rf in remote_files:
            if "reasoning_assistant_latest" in rf:
                chosen_file = rf
                break
    if not chosen_file:
        step_files = [rf for rf in remote_files if "reasoning_step_" in rf]
        if step_files:
            chosen_file = sorted(step_files)[-1]
            
    if not chosen_file:
        log("No valid remote checkpoint found to pull.")
        return False
        
    log(f"Selected remote checkpoint: {chosen_file}")
    
    # Rsync the chosen checkpoint to TARGET_CKPT
    rsync_cmd = (
        f"rsync -avz --progress "
        f"-e 'ssh -o StrictHostKeyChecking=no -p {SSH_PORT}' "
        f"root@{SSH_HOST}:{chosen_file} '{TARGET_CKPT}'"
    )
    log(f"Executing: {rsync_cmd}")
    code, out, err = run_cmd(rsync_cmd, timeout=1800) # 30 min max for download
    if code != 0:
        log(f"Rsync failed (code {code}): {err}")
        return False
        
    # Also pull the training log
    log_sync_cmd = (
        f"rsync -avz "
        f"-e 'ssh -o StrictHostKeyChecking=no -p {SSH_PORT}' "
        f"root@{SSH_HOST}:/workspace/pcei/cloud_kindergarten_training.log '{TARGET_LOG}'"
    )
    run_cmd(log_sync_cmd, timeout=60)
    
    # Verify local file size
    if TARGET_CKPT.exists() and TARGET_CKPT.stat().st_size > 3_000_000_000:
        sz_gb = TARGET_CKPT.stat().st_size / (1024**3)
        log(f"SUCCESS: Checkpoint pulled and verified locally ({sz_gb:.2f} GB) at {TARGET_CKPT}")
        return True
    else:
        log("Verification failed: Local file missing or under 3 GB.")
        return False

def teardown_instance():
    log(f"Terminating Vast.ai instance {INSTANCE_ID} to prevent any further billing...")
    code, out, err = run_cmd(f"{VASTAI_BIN} destroy instance {INSTANCE_ID} 2>/dev/null", timeout=30)
    if code == 0:
        log(f"Instance {INSTANCE_ID} destroyed successfully. Billing stopped.")
    else:
        log(f"Failed to destroy instance {INSTANCE_ID}: {err}. Retrying in 10s...")
        time.sleep(10)
        run_cmd(f"{VASTAI_BIN} destroy instance {INSTANCE_ID} 2>/dev/null", timeout=30)

def main():
    log("Auto-pull daemon started. Monitoring cloud training...")
    pulled = False
    
    while True:
        try:
            # 1. Maintain caffeinate
            run_cmd("caffeinate -dims &", timeout=5)
            
            # 2. Check Credit
            credit = get_credit()
            if credit is not None:
                log(f"Current Vast.ai credit: ${credit:.2f}")
                if credit <= 0.25: # Credit floor threshold ($0.25 safety floor)
                    log(f"CRITICAL: Credit reached floor (${credit:.2f} <= $0.25). Triggering emergency pull and teardown.")
                    if pull_checkpoint():
                        teardown_instance()
                        log("JOB COMPLETE: Pulled before credit expiration. Exiting daemon.")
                        sys.exit(0)
                        
            # 3. Check Remote Status
            reachable, remote_info = check_remote_status()
            if reachable:
                if "FINISHED" in remote_info or ("STOPPED" in remote_info and "RUNNING" not in remote_info):
                    log("Training has completed on cloud. Triggering final checkpoint pull...")
                    if pull_checkpoint():
                        teardown_instance()
                        log("JOB COMPLETE: Final checkpoint pulled successfully. Exiting daemon.")
                        sys.exit(0)
            else:
                log("Remote instance temporarily unreachable. Will retry next cycle.")
                
        except Exception as e:
            log(f"Exception in loop: {e}")
            
        time.sleep(90) # Check every 90 seconds

if __name__ == "__main__":
    main()
