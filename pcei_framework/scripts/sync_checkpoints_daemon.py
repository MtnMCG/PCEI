#!/usr/bin/env python3
"""
scripts/sync_checkpoints_daemon.py

Local background sync daemon that monitors the remote Vast.ai RTX 4090 host.
Whenever a 10k-step checkpoint or final checkpoint is completed, it downloads
the checkpoint to the local checkpoints directory without exhausting local disk space.
"""

import sys
import time
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOG_FILE = PROJECT_ROOT / "sync_daemon.log"

REMOTE_HOST = os.environ.get("REMOTE_HOST", "<REMOTE_GPU_IP>")
REMOTE_PORT = os.environ.get("REMOTE_PORT", "22")
REMOTE_USER = os.environ.get("REMOTE_USER", "root")
SSH_KEY = Path(os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")))
SIGNAL_FILE = "/workspace/pcei/checkpoints/latest_sync_ready.txt"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_local_free_gb() -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    return free / (1024 ** 3)


def check_remote_signal() -> str:
    cmd = [
        "ssh", "-i", str(SSH_KEY), "-p", str(REMOTE_PORT),
        "-o", "ConnectTimeout=8",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"cat {SIGNAL_FILE} 2>/dev/null || true"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        return res.stdout.strip()
    except Exception as e:
        return ""


def rsync_file(remote_name: str, local_name: str) -> bool:
    free_gb = get_local_free_gb()
    log(f"Local free space: {free_gb:.2f} GB")
    if free_gb < 3.5:
        log(f"[WARNING] Insufficient disk space ({free_gb:.2f} GB < 3.5 GB). Skipping download to prevent disk full.")
        return False

    remote_path = f"/workspace/pcei/checkpoints/{remote_name}"
    local_path = CHECKPOINTS_DIR / local_name

    log(f"Syncing {remote_path} -> {local_path} ...")
    cmd = [
        "rsync", "-avz", "--partial",
        "-e", f"ssh -i {SSH_KEY} -p {REMOTE_PORT}",
        f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}",
        str(local_path)
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode == 0:
        log(f"Successfully synced {local_name} in {elapsed:.1f}s!")
        return True
    else:
        log(f"[ERROR] Sync failed (code {res.returncode}): {res.stderr.strip()}")
        return False


def check_pipeline_finished() -> bool:
    cmd = [
        "ssh", "-i", str(SSH_KEY), "-p", str(REMOTE_PORT),
        "-o", "ConnectTimeout=8",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        "pgrep -f 'blended_instruction|test_extensive_generation|eval_hellaswag' > /dev/null && echo 'RUNNING' || (test -f /workspace/pipeline_all_finished.txt && echo 'FINISHED' || echo 'RUNNING')"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        return "FINISHED" in res.stdout
    except Exception:
        return False


def main():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("PCEI Local Checkpoint Sync Daemon Started.")
    log(f"Target: root@{REMOTE_HOST}:{REMOTE_PORT}")
    log(f"Destination: {CHECKPOINTS_DIR}")
    log("=" * 60)

    last_signal = ""

    while True:
        try:
            signal = check_remote_signal()
            if signal and signal != last_signal:
                log(f"New signal detected:\n{signal}")
                last_signal = signal

                # Parse signal
                is_final = "FINAL=True" in signal
                
                # Always sync latest checkpoint
                success = rsync_file("pcei_750m_reasoning_assistant_latest.pt", "pcei_750m_reasoning_assistant_latest.pt")
                if success:
                    # Maintain pcei_750m_reasoning_assistant.pt as hardlink/symlink to avoid 3.5GB duplication
                    final_target = CHECKPOINTS_DIR / "pcei_750m_reasoning_assistant.pt"
                    latest_target = CHECKPOINTS_DIR / "pcei_750m_reasoning_assistant_latest.pt"
                    try:
                        if final_target.exists() and not final_target.is_symlink():
                            final_target.unlink()
                        if not final_target.exists():
                            final_target.symlink_to(latest_target.name)
                            log(f"Linked {final_target.name} -> {latest_target.name}")
                    except Exception as e:
                        log(f"Link error: {e}")

                if is_final:
                    if check_pipeline_finished():
                        log("Final model confirmed and pipeline finished! All sync operations complete.")
                        break
                    else:
                        log("Signal marked FINAL, but remote pipeline is still actively training. Continuing sync daemon...")

        except KeyboardInterrupt:
            log("Sync daemon stopped by user.")
            break
        except Exception as e:
            log(f"Unexpected error in sync loop: {e}")

        time.sleep(45)


if __name__ == "__main__":
    main()
