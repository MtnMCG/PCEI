"""
PCEI WebSocket — Real-time metrics stream and chat relay.

Reads structured JSON from telemetry (latest.json) instead of regex-parsing logs.
Falls back to log regex only for legacy scripts that haven't been updated.
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Chat process management ──────────────────────────────────────────────────
_chat_proc = None
_chat_lock = asyncio.Lock()


@router.post("/api/chat/kill")
async def kill_chat():
    """Kill the active resonance_chat process and free its RAM."""
    global _chat_proc
    async with _chat_lock:
        if _chat_proc is not None and _chat_proc.returncode is None:
            try:
                _chat_proc.kill()
                await asyncio.wait_for(_chat_proc.wait(), timeout=1.0)
            except Exception:
                pass
            _chat_proc = None
            return {"status": "killed"}
        _chat_proc = None
        return {"status": "not_running"}


# ── Metrics WebSocket ────────────────────────────────────────────────────────

@router.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    last_mtime = 0.0
    last_log_pos = 0
    current_log_file = None

    try:
        while True:
            # Import here to avoid circular import at module level
            from dashboard.api.training import get_active_session

            session = get_active_session()

            if session:
                metrics_file = Path(session.get("metrics_file", ""))
                log_file = Path(session.get("log_file", ""))

                # ── Strategy 1: Read structured JSON (preferred) ─────────
                sent = False
                if metrics_file.exists():
                    try:
                        mtime = metrics_file.stat().st_mtime
                        if mtime > last_mtime:
                            last_mtime = mtime
                            data = json.loads(metrics_file.read_text())
                            data["source_type"] = "structured"
                            await websocket.send_json(data)
                            sent = True
                    except (json.JSONDecodeError, OSError):
                        pass

                # ── Strategy 2: Regex fallback for legacy scripts ────────
                if not sent and log_file.exists():
                    if str(log_file) != str(current_log_file):
                        current_log_file = str(log_file)
                        last_log_pos = 0

                    try:
                        with open(log_file, "r") as f:
                            f.seek(last_log_pos)
                            new_lines = f.readlines()
                            last_log_pos = f.tell()

                        if new_lines:
                            metrics = _parse_log_lines(new_lines)
                            if metrics:
                                metrics["source_type"] = "regex_fallback"
                                await websocket.send_json(metrics)
                    except OSError:
                        pass

            else:
                # No active session — send heartbeat
                last_mtime = 0.0
                last_log_pos = 0
                current_log_file = None
                await websocket.send_json({
                    "timestamp": datetime.now().isoformat(),
                    "type": "heartbeat",
                    "status": "idle",
                })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def _parse_log_lines(lines: list) -> Optional[dict]:
    """Legacy regex parser — fallback for scripts not yet using PCEITelemetry."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "step": 0,
        "loss": 0.0,
        "tokens_per_sec": 0.0,
        "throughput": 0.0,
        "active_bonds": 0,
        "lr": 0.0004,
        "learning_rate": 0.0004,
        "grad_norm": None,
        "eval_loss": None,
        "perplexity": None,
    }

    has_metrics = False
    for line in lines:
        m_step = re.search(r'(?:\[Step |step\s*:?\s*)(\d+)(?:\])?', line, re.IGNORECASE)
        if m_step:
            metrics["step"] = int(m_step.group(1))
            has_metrics = True

        m_loss = re.search(r'(?:Physics Loss \(MSE\):\s*|Loss:\s*|CE:\s*|loss\s*:?\s*)([0-9\.]+)', line, re.IGNORECASE)
        if m_loss:
            metrics["loss"] = float(m_loss.group(1))
            has_metrics = True

        m_tok = re.search(r'(?:Tok/s:\s*|tokens/sec:\s*|tok/s:\s*)([0-9\.]+)', line, re.IGNORECASE)
        if m_tok:
            val = float(m_tok.group(1))
            metrics["tokens_per_sec"] = val
            metrics["throughput"] = val
            has_metrics = True

        m_bonds = re.search(r'Active Bonds:\s*(\d+)', line, re.IGNORECASE)
        if m_bonds:
            metrics["active_bonds"] = int(m_bonds.group(1))
            has_metrics = True

        m_lr = re.search(r'(?:LR:\s*|lr:\s*|learning_rate:\s*)([0-9\.e\-\+]+)', line, re.IGNORECASE)
        if m_lr:
            try:
                val = float(m_lr.group(1))
                metrics["learning_rate"] = val
                metrics["lr"] = val
                has_metrics = True
            except ValueError:
                pass

        m_gnorm = re.search(r'(?:Grad(?:ient)?\s*Norm:\s*|GNorm:\s*)([0-9\.e\-\+]+)', line, re.IGNORECASE)
        if m_gnorm:
            try:
                metrics["grad_norm"] = float(m_gnorm.group(1))
                has_metrics = True
            except ValueError:
                pass

        m_eval = re.search(r'(?:Validation Loss:\s*|Eval Loss:\s*|eval_loss:\s*)([0-9\.]+)', line, re.IGNORECASE)
        if m_eval:
            try:
                metrics["eval_loss"] = float(m_eval.group(1))
                has_metrics = True
            except ValueError:
                pass

        m_ppl = re.search(r'Perplexity:\s*([0-9\.]+)', line, re.IGNORECASE)
        if m_ppl:
            try:
                metrics["perplexity"] = float(m_ppl.group(1))
                has_metrics = True
            except ValueError:
                pass

    return metrics if has_metrics else None


# ── Chat WebSocket ───────────────────────────────────────────────────────────

@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket, checkpoint: str = "pcei_100m.pt", mode: str = "inference"):
    """
    Each WebSocket connection gets exclusive ownership of the chat process.
    On disconnect, the process is killed to free RAM.
    mode='inference' skips training (fast). mode='learn' trains on input (slow).
    """
    global _chat_proc

    await websocket.accept()

    async with _chat_lock:
        # Kill any existing process — only one chat session at a time
        if _chat_proc is not None and _chat_proc.returncode is None:
            _chat_proc.terminate()
            try:
                await asyncio.wait_for(_chat_proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                _chat_proc.kill()
            _chat_proc = None

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PYTHONUNBUFFERED"] = "1"

        python_exe = str(PROJECT_ROOT / "venv" / "bin" / "python")
        if not Path(python_exe).exists():
            python_exe = sys.executable

        cmd = [
            python_exe,
            str(PROJECT_ROOT / "tools" / "resonance_chat.py"),
            "--api-mode",
            "--checkpoint-name", checkpoint,
        ]
        if mode == "inference":
            cmd.append("--inference-only")

        try:
            _chat_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except Exception as e:
            await websocket.send_text(f"[Error] Failed to launch model: {e}")
            await websocket.close()
            return

    proc = _chat_proc

    async def read_stdout():
        buf = ""
        try:
            while True:
                chunk = await proc.stdout.read(256)
                if not chunk:
                    break
                buf += chunk.decode(errors="replace")

                while buf:
                    si = buf.find("__START_MODEL__")
                    ei = buf.find("__END_MODEL__")

                    marker_pos = -1
                    marker_text = ""
                    marker_len = 0

                    if si >= 0 and (ei < 0 or si < ei):
                        marker_pos = si
                        marker_text = "__START_MODEL__"
                        marker_len = 15
                    elif ei >= 0:
                        marker_pos = ei
                        marker_text = "__END_MODEL__"
                        marker_len = 13

                    if marker_pos >= 0:
                        before = buf[:marker_pos]
                        if before.strip():
                            await websocket.send_text(before)
                        await websocket.send_text(marker_text)
                        buf = buf[marker_pos + marker_len:]
                        continue

                    if len(buf) > 15:
                        to_send = buf[:-15]
                        buf = buf[-15:]
                        if to_send:
                            await websocket.send_text(to_send)
                    else:
                        break

        except Exception:
            pass

        if buf.strip():
            try:
                await websocket.send_text(buf)
            except Exception:
                pass

        try:
            await websocket.send_text("\n[Session ended. Click Connect to start a new session.]")
        except Exception:
            pass

    stdout_task = asyncio.create_task(read_stdout())

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
            except asyncio.TimeoutError:
                break

            if proc.returncode is not None:
                break

            proc.stdin.write((data + "\n").encode())
            await proc.stdin.drain()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        stdout_task.cancel()
        async with _chat_lock:
            if proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
            if _chat_proc is proc:
                _chat_proc = None
