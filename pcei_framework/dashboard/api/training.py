"""
PCEI Training API — Session-persistent, single-slot training launcher.

Key design:
- One training job at a time (excluding resonance_chat).
- Session state persisted to disk (session.json) so any browser/server restart recovers.
- Smart kill: SIGTERM → 15s grace period for checkpoint save → SIGKILL.
- Structured log directory per run with separate output.log and metrics/ dir.
"""

import os
import sys
import re
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel
import json

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "dashboard" / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SESSION_FILE = LOGS_DIR / "session.json"

# ── In-memory state (recovered from disk on startup) ────────────────────────
_active_process: Optional[subprocess.Popen] = None
_active_session: Optional[dict] = None  # mirrors session.json


class LaunchRequest(BaseModel):
    script: str
    args: Dict[str, Any]


class KillRequest(BaseModel):
    pid: Optional[int] = None  # Optional — kills whatever is running

class DirectorRequest(BaseModel):
    prompt: str


SCRIPT_MAP = {
    'wiki_explorer': 'tools/wiki_explorer.py',
    'instruction_tuner': 'tools/instruction_tuner.py',
    'dreaming_chamber': 'tools/dreaming_chamber.py',
    'resonance_chat': 'tools/resonance_chat.py',
    'multimodal_tuner': 'tools/multimodal_tuner.py',
    'reading_room': 'tools/reading_room.py',
    'ollama_trainer': 'tools/ollama_trainer.py',
}


# ── Session Persistence ─────────────────────────────────────────────────────

def _save_session(session: dict):
    """Write session state to disk atomically."""
    global _active_session
    _active_session = session
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(session, indent=2))
        tmp.rename(SESSION_FILE)
    except Exception:
        pass


def _clear_session():
    """Remove session state from disk and memory."""
    global _active_session, _active_process
    _active_session = None
    _active_process = None
    try:
        SESSION_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_is_alive(pid: int) -> bool:
    """Check if a PID is still running and not a defunct zombie."""
    global _active_process
    if _active_process is not None and _active_process.pid == pid:
        return _active_process.poll() is None
    try:
        res = os.waitpid(pid, os.WNOHANG)
        if res != (0, 0):
            return False
        os.kill(pid, 0)
        return True
    except ChildProcessError:
        try:
            out = subprocess.check_output(["ps", "-o", "state=", "-p", str(pid)], text=True).strip()
            return bool(out) and not out.startswith("Z")
        except Exception:
            return False
    except (OSError, ProcessLookupError):
        return False


def _recover_session():
    """Called on module load. Recover session from disk if the process is still alive."""
    global _active_session
    if not SESSION_FILE.exists():
        return

    try:
        session = json.loads(SESSION_FILE.read_text())
    except Exception:
        _clear_session()
        return

    pid = session.get("pid")
    if pid and _pid_is_alive(pid):
        _active_session = session
        # We can't recover the Popen object, but we have the PID for kill/status
    else:
        _clear_session()


# Recover on import
_recover_session()


# ── Smart Kill ───────────────────────────────────────────────────────────────

def _smart_kill(pid: int, grace_seconds: float = 15.0) -> str:
    """
    SIGTERM → wait for graceful save → SIGKILL if needed.
    Returns: 'graceful', 'forced', or 'already_dead'
    """
    if not _pid_is_alive(pid):
        return "already_dead"

    # Step 1: SIGTERM (triggers KeyboardInterrupt in Python → checkpoint save)
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return "already_dead"

    # Step 2: Wait for graceful exit
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return "graceful"
        time.sleep(0.5)

    # Step 3: Force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    except (OSError, ProcessLookupError):
        pass

    return "forced"


# ── API Endpoints ────────────────────────────────────────────────────────────

@router.get("/training/status")
def get_training_status():
    """Return current training session status. Refresh-safe — reads from disk."""
    global _active_session

    # Re-check session validity
    if _active_session:
        pid = _active_session.get("pid")
        if pid and _pid_is_alive(pid):
            return {
                "is_running": True,
                "pid": pid,
                "script": _active_session.get("script"),
                "args": _active_session.get("args", {}),
                "start_time": _active_session.get("start_time"),
                "runtime_seconds": time.time() - _active_session.get("start_time", time.time()),
                "log_dir": _active_session.get("log_dir"),
                "log_file": _active_session.get("log_file"),
                "metrics_file": _active_session.get("metrics_file"),
            }
        else:
            # Process died on its own — clean up
            _clear_session()

    # Also try recovering from disk if in-memory is empty
    if not _active_session and SESSION_FILE.exists():
        _recover_session()
        if _active_session:
            return get_training_status()

    return {"is_running": False, "pid": None, "script": None}


@router.post("/training/launch")
def launch_training(req: LaunchRequest):
    """
    Launch a training script. Single-slot: kills any running job first.
    """
    global _active_process, _active_session

    if req.script not in SCRIPT_MAP:
        return {"error": f"Unknown script: {req.script}"}

    # ── Single-slot enforcement: kill existing job ───────────────────────
    if _active_session:
        old_pid = _active_session.get("pid")
        if old_pid and _pid_is_alive(old_pid):
            kill_result = _smart_kill(old_pid, grace_seconds=15.0)
        _clear_session()

    # ── Build command ────────────────────────────────────────────────────
    script_path = PROJECT_ROOT / SCRIPT_MAP[req.script]

    cmd_args = []
    for k, v in req.args.items():
        if v is None or str(v).strip() == "":
            continue
        cli_key = k.replace("_", "-")
        str_v = str(v).strip().lower()
        if isinstance(v, bool) or str_v in ("true", "false", "on", "off"):
            if v is True or str_v in ("true", "on"):
                cmd_args.append(f"--{cli_key}")
        else:
            cmd_args.extend([f"--{cli_key}", str(v)])

    # ── Create structured log directory ──────────────────────────────────
    timestamp = int(time.time())
    log_dir = LOGS_DIR / f"{req.script}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "output.log"
    metrics_dir = log_dir  # telemetry.py writes metrics.jsonl + latest.json here

    # ── Environment ──────────────────────────────────────────────────────
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PCEI_METRICS_DIR"] = str(metrics_dir)  # telemetry.py reads this
    env["PYTHONUNBUFFERED"] = "1"

    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    py_exec = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [py_exec, str(script_path)] + cmd_args

    try:
        f = open(log_file, "w")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        f.close()

        _active_process = proc

        session = {
            "pid": proc.pid,
            "script": req.script,
            "args": req.args,
            "start_time": time.time(),
            "log_dir": str(log_dir),
            "log_file": str(log_file),
            "metrics_file": str(metrics_dir / "latest.json"),
        }
        _save_session(session)

        return {"pid": proc.pid, "log_dir": str(log_dir)}

    except Exception as e:
        return {"error": str(e)}


@router.post("/training/kill")
def kill_training(req: KillRequest = None):
    """Smart kill: SIGTERM → checkpoint save → SIGKILL."""
    global _active_session

    if not _active_session:
        return {"error": "No active training session"}

    pid = _active_session.get("pid")
    if not pid:
        _clear_session()
        return {"error": "No PID in session"}

    result = _smart_kill(pid, grace_seconds=15.0)
    _clear_session()

    return {"status": result, "pid": pid}

@router.post("/training/director")
def send_director_prompt(req: DirectorRequest):
    """Write director prompt to file for ollama_trainer.py to pick up."""
    director_file = CHECKPOINTS_DIR / "director_prompt.txt"
    try:
        director_file.write_text(req.prompt)
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@router.get("/training/logs/{pid}")
def get_training_logs(pid: int):
    """Return recent log lines for the active or most recent session."""
    log_file = None

    # Check active session first
    if _active_session and _active_session.get("pid") == pid:
        log_file = Path(_active_session.get("log_file", ""))

    # Fallback: scan log dirs for this PID
    if not log_file or not log_file.exists():
        # Try to find a log dir that contains a process with this PID
        for d in sorted(LOGS_DIR.iterdir(), reverse=True):
            if d.is_dir():
                candidate = d / "output.log"
                if candidate.exists():
                    log_file = candidate
                    break

    if not log_file or not log_file.exists():
        return {"error": "Log file not found", "logs": []}

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            clean_lines = [l.rstrip() for l in lines[-300:]]
            return {"logs": clean_lines}
    except Exception as e:
        return {"error": str(e), "logs": []}


@router.get("/training/logs")
def get_current_logs():
    """Return logs for whatever is currently running (no PID needed)."""
    if not _active_session:
        return {"error": "No active session", "logs": []}
    log_file = Path(_active_session.get("log_file", ""))
    if not log_file.exists():
        return {"error": "Log file not found", "logs": []}
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            clean_lines = [l.rstrip() for l in lines[-300:]]
            return {"logs": clean_lines}
    except Exception as e:
        return {"error": str(e), "logs": []}


@router.get("/checkpoints")
def list_checkpoints():
    if not CHECKPOINTS_DIR.exists():
        return []

    ckpts = []
    for p in CHECKPOINTS_DIR.glob("*.pt"):
        file_size = os.path.getsize(p)
        mtime = os.path.getmtime(p)
        m = re.search(r'step[_\-]?(\d+)', p.name)
        global_step = int(m.group(1)) if m else 0

        ckpts.append({
            "filename": p.name,
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 1),
            "modified": mtime,
            "last_modified": mtime,
            "step": global_step,
            "global_step": global_step,
        })

    ckpts.sort(key=lambda x: x["last_modified"], reverse=True)
    return ckpts


# ── Expose session for websocket.py to import ───────────────────────────────

def get_active_session() -> Optional[dict]:
    """Used by websocket.py to find the active metrics file."""
    global _active_session
    if _active_session:
        pid = _active_session.get("pid")
        if pid and _pid_is_alive(pid):
            return _active_session
        else:
            _clear_session()
    return None
