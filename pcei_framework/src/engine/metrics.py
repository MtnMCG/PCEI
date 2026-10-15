"""
PCEI Metrics — Full Observability Stack

Logs EVERYTHING about the training run to JSON files that can be read 
in-IDE to diagnose issues, detect grokking, and guide refactoring.

Tracked metrics:
- Loss per step (total, per-modality)
- Ternary weight distributions {0, 1, 2} per layer over time
- slow_field_confidence per PlasticLayer over time
- Entanglement bond count and strength over time
- Topology positions (X,Y) for all PlasticLayers
- Gradient norms per layer
- dream_strength value over time
- ResponseGate settlement velocity
- Memory usage
"""
import json
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from collections import defaultdict
from datetime import datetime


from src.modules.ternary import TernaryQuantize, TernaryLinear
from src.modules.plasticity import PlasticLayer


class PCEIMetrics:
    """
    Central metrics collector for the PCEI training loop.
    Saves everything to JSON files readable by both humans and AI assistants.
    """
    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Step counter
        self.global_step = 0
        self.session_step = 0
        self.epoch = 0
        
        # Rolling loss history
        self.loss_history: List[Dict] = []
        
        # Per-layer tracking
        self.ternary_distributions: List[Dict] = []
        self.confidence_history: List[Dict] = []
        self.gradient_norms: List[Dict] = []
        self.topology_snapshots: List[Dict] = []
        
        # System-wide tracking
        self.bond_history: List[Dict] = []
        self.dream_strength_history: List[float] = []
        self.gate_velocity_history: List[Dict] = []
        
        # Grokking detection
        self.train_loss_window: List[float] = []
        self.eval_accuracy_window: List[float] = []
        
        # Timing
        self.step_times: List[float] = []
        self._step_start = None
        
        print(f"[PCEIMetrics] Logging to: {self.log_dir}")

    def step_start(self):
        self._step_start = time.time()

    def step_end(self):
        if self._step_start:
            self.step_times.append(time.time() - self._step_start)
        self.global_step += 1
        self.session_step += 1

    # ═══════════════════════════════════════
    # LOSS TRACKING
    # ═══════════════════════════════════════
    def log_loss(self, loss_total: float, loss_text: float = 0.0, 
                 loss_img: float = 0.0, loss_audio: float = 0.0):
        record = {
            "step": self.global_step,
            "session_step": self.session_step,
            "epoch": self.epoch,
            "loss_total": loss_total,
            "loss_text": loss_text,
            "loss_img": loss_img,
            "loss_audio": loss_audio,
            "timestamp": time.time()
        }
        self.loss_history.append(record)
        self.train_loss_window.append(loss_total)
        
        # Keep window manageable
        if len(self.train_loss_window) > 1000:
            self.train_loss_window = self.train_loss_window[-500:]

    def log_eval_accuracy(self, accuracy: float):
        self.eval_accuracy_window.append(accuracy)

    # ═══════════════════════════════════════
    # TERNARY WEIGHT DISTRIBUTIONS
    # ═══════════════════════════════════════
    def log_ternary_distributions(self, model: nn.Module):
        """Counts {0, 1, 2} values across all TernaryLinear layers."""
        snapshot = {"step": self.global_step, "session_step": self.session_step, "layers": {}}
        
        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                with torch.no_grad():
                    q = TernaryQuantize.apply(module.weight, module.threshold_low, module.threshold_high)
                    total = q.numel()
                    count_0 = (q == 0).sum().item()
                    count_1 = (q == 1).sum().item()
                    count_2 = (q == 2).sum().item()
                    
                    snapshot["layers"][name] = {
                        "pruned_0": count_0 / total,
                        "partial_1": count_1 / total,
                        "strong_2": count_2 / total,
                        "total_params": total
                    }
        
        self.ternary_distributions.append(snapshot)

    # ═══════════════════════════════════════
    # PLASTICITY & CONFIDENCE
    # ═══════════════════════════════════════
    def log_plasticity(self, model: nn.Module):
        """Logs slow_field_confidence and effective_lr for every PlasticLayer."""
        snapshot = {"step": self.global_step, "session_step": self.session_step, "layers": {}}
        
        for name, module in model.named_modules():
            if isinstance(module, PlasticLayer):
                conf = module.slow_field_confidence.item()
                pos = module.get_position()
                
                snapshot["layers"][module.layer_id] = {
                    "confidence": conf,
                    "effective_lr_factor": 1.0 - conf,
                    "pos_x": pos[0],
                    "pos_y": pos[1],
                    "pos_z": pos[2]
                }
        
        self.confidence_history.append(snapshot)

    # ═══════════════════════════════════════
    # GRADIENT NORMS
    # ═══════════════════════════════════════
    def log_gradient_norms(self, model: nn.Module):
        """Logs gradient L2 norm per PlasticLayer (call after backward())."""
        snapshot = {"step": self.global_step, "session_step": self.session_step, "layers": {}}
        
        for name, module in model.named_modules():
            if isinstance(module, PlasticLayer):
                total_norm = 0.0
                param_count = 0
                for p in module.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        param_count += 1
                total_norm = total_norm ** 0.5
                
                snapshot["layers"][module.layer_id] = {
                    "grad_norm": total_norm,
                    "param_count": param_count
                }
        
        self.gradient_norms.append(snapshot)

    # ═══════════════════════════════════════
    # TOPOLOGY POSITIONS
    # ═══════════════════════════════════════
    def log_topology(self, model: nn.Module):
        """Captures X,Y,Z positions of all PlasticLayers for migration visualization."""
        snapshot = {"step": self.global_step, "session_step": self.session_step, "positions": {}}
        
        for name, module in model.named_modules():
            if isinstance(module, PlasticLayer):
                pos = module.get_position()
                snapshot["positions"][module.layer_id] = {
                    "x": pos[0], "y": pos[1], "z": pos[2]
                }
        
        self.topology_snapshots.append(snapshot)

    # ═══════════════════════════════════════
    # ENTANGLEMENT BONDS
    # ═══════════════════════════════════════
    def log_bonds(self, active_bonds: Dict):
        record = {
            "step": self.global_step,
            "session_step": self.session_step,
            "bond_count": len(active_bonds),
            "bonds": {f"{a}-{b}": strength for (a, b), strength in active_bonds.items()},
            "mean_strength": sum(active_bonds.values()) / max(len(active_bonds), 1)
        }
        self.bond_history.append(record)

    # ═══════════════════════════════════════
    # DREAM STRENGTH
    # ═══════════════════════════════════════
    def log_dream_strength(self, dream_strength: float):
        self.dream_strength_history.append(dream_strength)

    # ═══════════════════════════════════════
    # RESPONSE GATE
    # ═══════════════════════════════════════
    def log_gate_velocity(self, velocity: float, settled: bool):
        self.gate_velocity_history.append({
            "step": self.global_step,
            "session_step": self.session_step,
            "velocity": velocity,
            "settled": settled
        })

    # ═══════════════════════════════════════
    # GROKKING DETECTION
    # ═══════════════════════════════════════
    def detect_grokking(self, window: int = 100) -> Dict[str, Any]:
        """
        Grokking = loss plateaus/decreases slowly while accuracy suddenly jumps.
        Returns a signal dict indicating whether grokking might be occurring.
        """
        if len(self.train_loss_window) < window or len(self.eval_accuracy_window) < 2:
            return {"grokking_detected": False, "reason": "insufficient_data"}
        
        recent_loss = self.train_loss_window[-window:]
        loss_trend = (sum(recent_loss[-window//2:]) / (window//2)) - \
                     (sum(recent_loss[:window//2]) / (window//2))
        
        if len(self.eval_accuracy_window) >= 2:
            acc_jump = self.eval_accuracy_window[-1] - self.eval_accuracy_window[-2]
        else:
            acc_jump = 0.0
        
        # Grokking signal: Emergent Mutual Information Jump
        # We look for accuracy improving significantly more than the loss would suggest.
        # If loss_trend is near zero (plateau) but accuracy is trending upward, it's a signal.
        if len(self.eval_accuracy_window) >= 5:
            acc_trend = self.eval_accuracy_window[-1] - self.eval_accuracy_window[-5]
        else:
            acc_trend = acc_jump
            
        # Detect: Loss is stagnant (> -0.05) AND accuracy is moving (+0.01)
        grokking = (loss_trend > -0.05) and (acc_trend > 0.01)
        
        return {
            "grokking_detected": grokking,
            "loss_trend": loss_trend,
            "accuracy_jump": acc_jump,
            "recent_loss_mean": sum(recent_loss) / len(recent_loss),
            "latest_accuracy": self.eval_accuracy_window[-1] if self.eval_accuracy_window else None
        }

    # ═══════════════════════════════════════
    # SAVE / EXPORT
    # ═══════════════════════════════════════
    def save_snapshot(self, tag: str = ""):
        """Saves all accumulated metrics to JSON files.
        Always saves BOTH tagged and untagged versions so the plotter can find them."""
        suffix = f"_{tag}" if tag else ""
        
        def _save(data, name):
            # Always save the canonical (untagged) version
            path = self.log_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            # Also save tagged version if tag provided
            if suffix:
                tagged_path = self.log_dir / f"{name}{suffix}.json"
                with open(tagged_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        _save(self.loss_history, "loss_history")
        _save(self.ternary_distributions, "ternary_distributions")
        _save(self.confidence_history, "confidence_history")
        _save(self.gradient_norms, "gradient_norms")
        _save(self.topology_snapshots, "topology_snapshots")
        _save(self.bond_history, "bond_history")
        _save({"dream_strength": self.dream_strength_history}, "dream_strength")
        _save(self.gate_velocity_history, "gate_velocity")
        _save({"step_times": self.step_times}, "timing")
        _save(self.detect_grokking(), "grokking_status")
        
        # Also save a compact summary CSV for quick scanning
        csv_path = self.log_dir / f"loss_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss_total", "loss_text", "loss_img", "loss_audio"])
            for record in self.loss_history:
                writer.writerow([
                    record["step"], record["epoch"],
                    f"{record['loss_total']:.6f}",
                    f"{record['loss_text']:.6f}",
                    f"{record['loss_img']:.6f}",
                    f"{record['loss_audio']:.6f}"
                ])
        
        print(f"[PCEIMetrics] Snapshot saved to {self.log_dir}/")

    def get_summary(self) -> Dict[str, Any]:
        """Returns a compact dict of the current state for quick inspection."""
        recent_losses = self.loss_history[-10:] if self.loss_history else []
        avg_loss = sum(r["loss_total"] for r in recent_losses) / max(len(recent_losses), 1)
        
        return {
            "global_step": self.global_step,
            "session_step": self.session_step,
            "epoch": self.epoch,
            "avg_recent_loss": avg_loss,
            "total_bonds": self.bond_history[-1]["bond_count"] if self.bond_history else 0,
            "dream_strength": self.dream_strength_history[-1] if self.dream_strength_history else 0,
            "grokking": self.detect_grokking(),
            "avg_step_time": sum(self.step_times[-100:]) / max(len(self.step_times[-100:]), 1) if self.step_times else 0,
        }
