# Matched-Parameter Ablation: FastKANMixer vs. MLP

**Date:** 2026-10-15  
**Author:** Independent external audit (Muse, Meta)  
**Code under test:** `pcei_framework/src/modules/adaptation.py` → `FastKANMixer`, at PCEI commit `ce27389` (2026-10-15 13:22 PDT)  
**Environment:** CPU-only (2 vCPU, 7.7 GB RAM), PyTorch 2.14.1+cpu  

---

## 1. Research question

Does PCEI's `FastKANMixer` (per-channel learnable cubic polynomial + LayerNorm) outperform a plain MLP at matched parameter capacity on nonlinear function fitting — or is it architectural decoration? This is the small-scale, compute-cheap version of the standard skeptical objection to KAN-style modules ("KANs vs. MLP at matched FLOPs or it didn't happen").

## 2. Experimental design

**Task — per-channel univariate regression.** Each of C = 16 channels is assigned a fixed nonlinear scalar function $f_c(x)$, with $x \sim U(-2, 2)$. Both contenders are drop-in $(B, L, C) \to (B, L, C)$ blocks trained to reproduce $f_c(x)$ per channel. The eight target functions, each assigned to two channels:

| # | Target function |
|---|---|
| 1 | $\sin(2x) + 0.3x$ |
| 2 | $0.5x^3 - x$ |
| 3 | $x|x|$ |
| 4 | $\tanh(3x)$ |
| 5 | $2 \cdot \exp(-x^2)$ |
| 6 | $x \cdot \sin(3x)$ |
| 7 | $0.25x^2 + 0.75 \cdot \cos(1.5x)$ |
| 8 | $2 \cdot \text{sigmoid}(2x) - 1$ |

**Contenders (parameter counts bracket the module under test):**

| Model | Architecture | Params |
|---|---|---|
| FastKANMixer | per-channel learnable cubic polynomial + LayerNorm (degree 3) | 96 |
| MLP h=2 | Linear(16→2) → ReLU → Linear(2→16) | 82 |
| MLP h=3 | Linear(16→3) → ReLU → Linear(3→16) | 115 |

**Protocol.** Identical for all contenders: 6 random seeds, 1500 full-batch Adam steps (lr = 1e-3), 2048 samples × 32 positions × 16 channels per seed, MSE loss. No per-model hyperparameter tuning. Metric: final training MSE, mean ± std over seeds. Wall-clock per run recorded as a compute proxy.

## 3. Results

| Model | Params | Final MSE (mean ± std, 6 seeds) | Sec / run |
|---|---|---|---|
| **FastKANMixer (deg 3)** | 96 | **0.1079 ± 0.0004** | 45.1 |
| MLP h=2 | 82 | 0.5653 ± 0.0061 | 13.3 |
| MLP h=3 | 115 | 0.5260 ± 0.0036 | 15.5 |

All runs completed with finite outputs (no NaN/Inf). The FastKANMixer achieves **~5× lower final MSE** than either MLP at matched parameter count, with an order of magnitude tighter variance across seeds (0.0004 vs. 0.004–0.006).

## 4. Limitations

1. **Toy task, favorable structure.** Per-channel univariate regression is structurally favorable to per-channel polynomial modules. This is a mechanism check on the module in isolation, not evidence that FastKANMixer improves end-to-end performance inside PCEI.
2. **Shared MLP bottleneck.** The MLP contenders share a single narrow hidden layer across all 16 channels, which bottlenecks them by design. This is the capacity tradeoff under test, but it should be stated plainly.
3. **Compute cost.** FastKANMixer was ~3× slower per training step (45 s vs. 13–15 s per run) due to an unfused Python loop over polynomial degrees. An einsum/Horner's-method rewrite would be expected to close most of this gap; inference cost was not measured.
4. **No tuning.** Learning rate and step budget were fixed identically for all models; the MLPs were not tuned. Results reflect out-of-the-box behavior, not each architecture's best case.

## 5. Conclusion

At matched parameter counts on a fixed nonlinear regression task, PCEI's FastKANMixer fits substantially better and more consistently than a plain MLP. This is the first matched-capacity data point for the module and it favors the PCEI design — with the caveat that it is a small-scale mechanism check, not a claim about full-system performance. Recommended follow-ups: (a) fuse the polynomial evaluation loop and re-measure compute cost; (b) repeat the comparison inside a trained PCEI stack (module swap ablation) rather than in isolation.

## 6. Reproducibility

Script: `fastkan_vs_mlp.py` · Results: `fastkan_vs_mlp_results.json`

```bash
python fastkan_vs_mlp.py   # writes fastkan_vs_mlp_results.json
```

Requires the PCEI repository on `sys.path` (`pcei_framework/src`) for `modules.adaptation.FastKANMixer`, plus PyTorch. Deterministic given seeds; rerun to verify.

---

## 7. PCEI Core Team Follow-Up & Resolution (2026-10-15)

In direct response to Limitation #3 and Recommendation (a), the PCEI team implemented **Horner's Polynomial Evaluation** directly in `pcei_framework/src/modules/adaptation.py`:

```python
# Horner's method: P(x) = c_0 + x * (c_1 + x * (c_2 + ...))
# Eliminates high-degree power calculations and redundant intermediate allocations
x_clamped = torch.clamp(x_norm, -2.0, 2.0)
out = self.poly_coeffs[:, self.degree].unsqueeze(0).unsqueeze(0).expand_as(x_clamped)
for d in range(self.degree - 1, -1, -1):
    c = self.poly_coeffs[:, d].unsqueeze(0).unsqueeze(0)
    out = c + x_clamped * out
```

### Verification & Benchmark Results:
- **Numerical Parity:** $\max |y_{\text{Horner}} - y_{\text{unfused}}| = 2.38 \times 10^{-7}$ (exact float32 machine epsilon parity).
- **Execution Speed:** Forward pass latency dropped from **$1,226\text{ ms} \to 532\text{ ms}$** ($2.30\times$ speedup).
- **Test Suite:** `tests/test_adaptation.py` passing 100%.
