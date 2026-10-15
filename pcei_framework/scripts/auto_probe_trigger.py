#!/usr/bin/env python3
"""
scripts/auto_probe_trigger.py

Automated trigger daemon for PCEI Multimodal Synergism Probe.
Monitors training until Phase 2 (Multimodal Integration) finishes at Step 88,251,
instantly snapshots the post-multimodal checkpoint, and runs the comparative probe.
"""

import time
import os
import shutil
import subprocess
from pathlib import Path

TRAIN_LOG = Path("/workspace/train.log")
CKPT_DIR = Path("/workspace/pcei/checkpoints")
PROBE_SCRIPT = Path("/workspace/pcei/scripts/probe_multimodal_synergy.py")
TARGET_STEP = 88251

print(f"[AutoProbeTrigger] Monitoring {TRAIN_LOG} for Step {TARGET_STEP} or 'Integration phase complete'...", flush=True)

while True:
    if TRAIN_LOG.exists():
        try:
            with open(TRAIN_LOG, "r", errors="ignore") as f:
                # Read last 50KB to keep it fast
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 50000))
                content = f.read()

            if ("Integration phase complete" in content) or (f"Step {TARGET_STEP}/" in content) or ("Phase 3: Internal Dialogue" in content):
                print(f"[AutoProbeTrigger] Phase 2 Multimodal completion detected at {time.strftime('%X')}! Triggering probe...", flush=True)
                # Wait 5 seconds for checkpoint flush
                time.sleep(5)
                
                supercharged = CKPT_DIR / "pcei_750m_supercharged.pt"
                step88251_copy = CKPT_DIR / "pcei_750m_step88251_multimodal_completed.pt"
                if supercharged.exists():
                    shutil.copy2(supercharged, step88251_copy)
                    print(f"[AutoProbeTrigger] Checkpoint archived to {step88251_copy} ({step88251_copy.stat().st_size / 1e9:.2f} GB)", flush=True)
                
                baseline = CKPT_DIR / "pcei_750m_step85251_text_baseline.pt"
                target_multi = step88251_copy if step88251_copy.exists() else supercharged
                
                cmd = [
                    "python3", str(PROBE_SCRIPT),
                    "--baseline", str(baseline),
                    "--multimodal", str(target_multi),
                    "--device", "cuda"
                ]
                print(f"[AutoProbeTrigger] Executing: {' '.join(cmd)}", flush=True)
                with open("/workspace/probe_results.log", "w") as out_f:
                    subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONPATH="/workspace/pcei"),
                        check=False
                    )
                print(f"[AutoProbeTrigger] Probe execution complete! Results saved to /workspace/probe_results.log", flush=True)
                break
        except Exception as e:
            print(f"[AutoProbeTrigger] Error checking log: {e}", flush=True)
            
    time.sleep(10)
