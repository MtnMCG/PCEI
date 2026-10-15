import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class AmbientField:
    """
    The dual-timescale subconscious field (Spec v0.1 Section 5).
    Captures activation history to prime the DreamSeeder and prevent catastrophic forgetting.

    Per-layer confidence is computed as cosine similarity between each layer's
    recent activation EMA and the consolidated slow_state. Layers whose activations
    are stable and aligned with consolidated memory get high confidence (low effective_lr).
    Volatile or newly-expanded layers get low confidence (high effective_lr).
    """
    def __init__(self, model: nn.Module, fast_decay: float = 0.9, slow_decay: float = 0.99):
        self.model = model
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay

        # Dual-Timescale Latent States (The 'Field')
        self.fast_state = None
        self.slow_state = None

        # Per-layer activation EMAs: {layer_id: torch.Tensor (d_model,)}
        self.layer_emas: Dict[str, torch.Tensor] = {}
        self._layer_ema_decay = 0.95

    def tick(self, latent: torch.Tensor, loss: float,
             detectors: Optional[Dict] = None):
        """
        Captures activations on every forward pass.
        Integrates current experience into the field.
        Updates per-layer confidence from activation-vs-slow_state similarity.

        Args:
            latent: Text stack output (batch, seq, d_text) -- used for field state
            loss: Scalar training loss for this step
            detectors: Optional dict of {layer_id: ResonanceDetector} -- used for
                       per-layer confidence. Pass nexus.core.detectors from the
                       training loop.
        """
        from src.modules.plasticity import PlasticLayer
        with torch.no_grad():
            # Shield: Remove NaNs before they infect the field
            latent_clean = torch.nan_to_num(latent.detach(), nan=0.0)
            # Compress latent to a representative vector (mean across sequence)
            compressed = latent_clean.mean(dim=1).cpu()

            # 1. Update Fast State (Working Memory)
            if self.fast_state is None:
                self.fast_state = compressed
            else:
                self.fast_state = (self.fast_state * self.fast_decay) + (compressed * (1.0 - self.fast_decay))

            # 2. Update Slow State (Consolidated Memory)
            if self.slow_state is None:
                self.slow_state = self.fast_state.clone()
            else:
                self.slow_state = (self.slow_state * self.slow_decay) + (self.fast_state * (1.0 - self.slow_decay))

            # 3. Per-layer confidence update
            # Strategy A: if detectors are provided, compute true per-layer confidence
            # from activation-vs-slow_state cosine similarity.
            # Strategy B: fallback to mild global nudge for tools that don't pass detectors.
            if detectors and self.slow_state is not None:
                self._update_per_layer_confidence_from_detectors(detectors)
            elif torch.rand(1).item() < 0.1:
                safe_loss = float(loss) if not (math.isnan(loss) or math.isinf(loss)) else 10.0
                global_target = 1.0 / (1.0 + safe_loss)
                for name, module in self.model.named_modules():
                    if isinstance(module, PlasticLayer):
                        conf = module.slow_field_confidence.item()
                        # Very mild nudge so it doesn't collapse differentiation
                        module.slow_field_confidence.copy_(
                            torch.tensor([conf * 0.999 + global_target * 0.001])
                        )

    def _update_per_layer_confidence_from_detectors(self, detectors: Dict):
        """
        Computes true per-layer confidence as cosine similarity between:
          - The layer's activation EMA (what it's currently doing)
          - The slow_state (what consolidated memory looks like)

        High similarity -> layer is stable, aligned with memory -> high confidence -> low lr.
        Low similarity -> layer is volatile or newly expanded -> low confidence -> high lr.
        """
        from src.modules.plasticity import PlasticLayer

        slow_ref = self.slow_state.squeeze().cpu()  # (d_slow,)

        for layer_id, detector in detectors.items():
            if len(detector.activation_buffer) == 0:
                continue

            # Latest activation, already mean-pooled to (d_model,) by register_activation
            latest_act = detector.activation_buffer[-1].cpu()

            # Update per-layer EMA
            if layer_id not in self.layer_emas:
                self.layer_emas[layer_id] = latest_act.clone()
            else:
                self.layer_emas[layer_id] = (
                    self.layer_emas[layer_id] * self._layer_ema_decay
                    + latest_act * (1.0 - self._layer_ema_decay)
                )

            layer_vec = self.layer_emas[layer_id]

            # Align dimensions: crop to shared prefix for cosine comparison
            d_min = min(layer_vec.shape[-1], slow_ref.shape[-1])
            layer_cmp = layer_vec[:d_min]
            slow_cmp = slow_ref[:d_min]

            layer_norm = layer_cmp.norm() + 1e-8
            slow_norm = slow_cmp.norm() + 1e-8

            # Cosine similarity in [-1, 1] -> confidence in [0, 1]
            cos_sim = (layer_cmp * slow_cmp).sum() / (layer_norm * slow_norm)
            confidence = ((cos_sim + 1.0) / 2.0).clamp(0.0, 1.0).item()

            if math.isnan(confidence):
                confidence = 0.0

            # Find the PlasticLayer by layer_id and apply slow EMA update
            for name, module in self.model.named_modules():
                if isinstance(module, PlasticLayer) and module.layer_id == layer_id:
                    old_conf = module.slow_field_confidence.item()
                    new_conf = old_conf * 0.98 + confidence * 0.02
                    module.slow_field_confidence.copy_(torch.tensor([new_conf]))
                    break

    def tick_consolidation(self):
        """Compatibility shim for old scripts/train.py logic."""
        pass

    def get_dream_data(self) -> torch.Tensor:
        """Returns the current consolidated state to shape DreamSeeder noise."""
        if self.slow_state is not None:
            return self.slow_state
        return torch.zeros(1, 128)
