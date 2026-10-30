#!/usr/bin/env python3
"""
PCEI dependency bootstrap (idempotent, self-repairing, shared by every launcher).

Guarantees that `import torch, tiktoken, huggingface_hub` works, then (NVIDIA machines only)
tries to swap in a CUDA build of PyTorch. If no CUDA build works for this Python/driver it
RESTORES the CPU build, so the environment is never left without PyTorch.
Exit code 0 = ready to run, 1 = could not produce a working environment.
"""
import os
import shutil
import subprocess
import sys

PIP = [sys.executable, "-m", "pip"]
CUDA_TAGS = ["cu126", "cu124", "cu121"]
BASE = ["torch", "tiktoken", "huggingface_hub", "numpy", "pyyaml"]


def run(args, quiet=False):
    sink = subprocess.DEVNULL if quiet else None
    return subprocess.call(args, stdout=sink, stderr=sink)


def imports_ok():
    return run([sys.executable, "-c", "import torch, tiktoken, huggingface_hub"], quiet=True) == 0


def cuda_ok():
    return run([sys.executable, "-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"], quiet=True) == 0


def has_nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        win = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = win if os.path.exists(win) else None
    return bool(exe) and run([exe], quiet=True) == 0


def install_base():
    for attempt in (1, 2):
        print(f"[bootstrap] Installing core packages (attempt {attempt}/2)...", flush=True)
        run(PIP + ["install", "--upgrade"] + BASE)
        if imports_ok():
            return True
    return False


def try_cuda():
    for tag in CUDA_TAGS:
        print(f"[bootstrap] NVIDIA GPU detected - trying CUDA PyTorch ({tag})...", flush=True)
        run(PIP + ["uninstall", "-y", "torch"], quiet=True)
        rc = run(PIP + ["install", "torch", "--index-url", f"https://download.pytorch.org/whl/{tag}"])
        if rc == 0 and imports_ok() and cuda_ok():
            return tag
    return None


def main():
    print(f"[bootstrap] Python {sys.version.split()[0]} on {sys.platform}", flush=True)

    if not imports_ok():
        if not install_base():
            print("[bootstrap] ERROR: could not install torch/tiktoken/huggingface_hub. "
                  "Check your internet connection, then re-run. (Tip: delete the 'venv' folder "
                  "and relaunch for a clean install; Python 3.10-3.12 is the most compatible.)", flush=True)
            return 1

    if has_nvidia() and not cuda_ok():
        tag = try_cuda()
        if tag:
            print(f"[bootstrap] GPU acceleration enabled (CUDA {tag}).", flush=True)
        else:
            print("[bootstrap] No CUDA build of PyTorch worked for this Python/driver "
                  "(an older Python such as 3.12 and an up-to-date NVIDIA driver usually fix this). "
                  "Restoring the CPU build so PCEI still runs...", flush=True)
            run(PIP + ["install", "--force-reinstall", "torch"])
            if not imports_ok() and not install_base():
                print("[bootstrap] ERROR: environment is broken and could not be repaired.", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
