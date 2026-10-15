"""
PCEI Telemetry — Structured metrics emitter for all training scripts.

Usage in any training script:
    from dashboard.telemetry import PCEITelemetry
    
    telem = PCEITelemetry.from_env()  # reads PCEI_METRICS_FILE env var
    # or: telem = PCEITelemetry("/path/to/metrics.jsonl")
    
    telem.emit(step=42, loss=2.31, grad_norm=0.41, ...)

Writes one JSON line per emit() call to a .jsonl file (append-mode).
Also maintains a latest.json for the WebSocket to poll efficiently.
"""

import json
import os
import time
import signal
import sys
from pathlib import Path


class PCEITelemetry:
    """Atomic structured metrics writer for PCEI training scripts."""

    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.metrics_dir / "metrics.jsonl"
        self.latest_path = self.metrics_dir / "latest.json"
        self._jsonl_file = open(self.jsonl_path, "a", buffering=1)  # line-buffered
        self._shutdown_requested = False
        self._original_sigterm = None

    @classmethod
    def from_env(cls) -> "PCEITelemetry":
        """Create from PCEI_METRICS_DIR environment variable (set by training.py launcher)."""
        metrics_dir = os.environ.get("PCEI_METRICS_DIR")
        if not metrics_dir:
            # Fallback: create in dashboard/logs/fallback_<pid>
            fallback = Path(__file__).parent / "logs" / f"fallback_{os.getpid()}"
            fallback.mkdir(parents=True, exist_ok=True)
            metrics_dir = str(fallback)
        return cls(metrics_dir)

    def emit(self, step: int, loss: float, **kwargs):
        """
        Emit a single metrics snapshot. Called once per training step.
        
        Required: step, loss
        Optional kwargs: eval_loss, perplexity, grad_norm, tokens_per_sec, lr,
                         active_bonds, bonds_formed_total, bonds_dissolved_total,
                         fast_slow_divergence, mean_confidence, epoch, source
        """
        data = {
            "step": step,
            "loss": round(loss, 6) if loss is not None else None,
            "timestamp": time.time(),
        }

        # Standard optional fields with rounding
        float_fields = [
            "eval_loss", "perplexity", "grad_norm", "tokens_per_sec",
            "lr", "fast_slow_divergence", "mean_confidence",
        ]
        int_fields = [
            "active_bonds", "bonds_formed_total", "bonds_dissolved_total",
            "epoch",
        ]

        for key in float_fields:
            val = kwargs.get(key)
            if val is not None:
                try:
                    data[key] = round(float(val), 6)
                except (TypeError, ValueError):
                    pass

        for key in int_fields:
            val = kwargs.get(key)
            if val is not None:
                try:
                    data[key] = int(val)
                except (TypeError, ValueError):
                    pass

        # Pass through any extra string fields (e.g. source, mode, dream_text, seed_topic)
        for key in ("source", "mode", "script", "dream_text", "seed_topic"):
            val = kwargs.get(key)
            if val is not None:
                data[key] = str(val)

        line = json.dumps(data)

        # Append to JSONL (line-buffered, always flushed)
        try:
            self._jsonl_file.write(line + "\n")
        except Exception:
            pass

        # Atomic overwrite of latest.json (write tmp → rename)
        try:
            tmp = self.latest_path.with_suffix(".tmp")
            tmp.write_text(line)
            tmp.rename(self.latest_path)
        except Exception:
            pass

    def install_graceful_shutdown(self, save_callback=None):
        """
        Install SIGTERM handler that:
        1. Sets shutdown_requested flag (training loop checks this)
        2. Optionally calls save_callback for checkpoint saving
        3. Exits cleanly
        
        Usage:
            def save_fn():
                torch.save(...)
            telem.install_graceful_shutdown(save_callback=save_fn)
            
            # In training loop:
            if telem.shutdown_requested:
                break
        """
        self._save_callback = save_callback

        def _handle_sigterm(signum, frame):
            self._shutdown_requested = True
            # Also raise KeyboardInterrupt so existing try/except handlers work
            raise KeyboardInterrupt("SIGTERM received — graceful shutdown")

        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def close(self):
        """Flush and close the JSONL file."""
        try:
            self._jsonl_file.close()
        except Exception:
            pass
        # Restore original signal handler
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except Exception:
                pass
