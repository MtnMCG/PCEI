# Tiny LM Bake-Off: PCEI-tiny vs Transformer vs Dense Mamba

**Date:** 2026-10-15  
**Author:** Independent external audit (Muse, Meta)  
**Code under test:** `pcei_framework/src/modules/mamba.py` (`MambaBlock`), `pcei_framework/src/modules/ternary.py` (`TernaryLinear`), `pcei_framework/src/modules/adaptation.py` (`FastKANMixer`), at PCEI commit `efd29f7` (2026-10-15)  
**Environment:** CPU-only (2 vCPU, 7.7 GB RAM), PyTorch 2.14.1+cpu  

---

## 1. Research question

Does a PCEI-style text stack (selective SSM + strict ternary quantization + FastKAN mixing) match or beat a plain transformer and a plain dense Mamba at matched parameter counts — or does aggressive quantization destroy the model? This is the small-scale, compute-cheap version of the "compute-matched baseline" objection: no frontier budget required, but the methodology (same data, same budget, same optimizer) is the same one a scaling curve would use.

---

## 2. Experimental design

**Models (~400k params each, matched as closely as possible):**

| Model | Architecture | Params |
| :--- | :--- | :---: |
| **Transformer** | Vanilla decoder: $d=96$, 5 layers, 4 heads, FFN $2\times d$ GELU, pre-norm, learned embeddings, causal mask | 398,688 |
| **Mamba (dense)** | Repo `MambaBlock` ($d_{\text{inner}}=192$, $d_{\text{state}}=8$, $d_{\text{conv}}=4$, unidirectional), 6 layers, pre-norm residuals. No quantization, no KAN, no ambient extras | 407,232 |
| **PCEI-tiny** | Same as Mamba, but every `nn.Linear` inside each block swapped post-construction for `TernaryLinear` (24 linears; quantized values verified strictly $\{-1, 0, +1\}$, STE gradients finite), plus `FastKANMixer(channels=96, degree=3)` as a second residual branch per block | 411,912 |

Embeddings and the LM head stay dense on all three (standard BitNet practice). `PCEI-tiny` is a representative composition, not the full `PCEINexus`: no diffusion steps, no ambient field, no neurogenesis, no response gating. Results speak to (ternary SSM + FastKAN) in isolation.

**Task — synthetic next-token prediction.** Generated in-code per seed. Vocab 128, seq len $48 = 6$ segments of 8 tokens. Each segment: key $k \in [1..16]$, 6 payload tokens from a key-dependent multinomial over tokens 32–127 (Dirichlet-randomized per seed), checksum token $(7k + \sum\text{payload}) \bmod 16 + 112$. 10% uniform token corruption as noise. 4,096 train / 512 val sequences. The key $\to$ payload statistics are locally learnable; the checksum forces a fixed 7-step memory dependency.

**Protocol (identical for all three).** AdamW, $\text{lr}=3\times 10^{-4}$, weight decay 0.01, batch 16, 600 steps (~2.3 epochs), seeds $\{0, 1\}$. No per-model tuning. Metrics: final train loss, val loss, val perplexity, forward tokens/sec on CPU, wall-clock minutes per run.

---

## 3. Results

**Per-seed (val loss):**

| Seed | Transformer | Mamba (dense) | PCEI-tiny |
| :---: | :---: | :---: | :---: |
| 0 | 4.6133 (ppl 100.8) | 4.3890 (ppl 80.6) | **4.3856 (ppl 80.3)** |
| 1 | 4.6128 (ppl 100.8) | 4.3872 (ppl 80.4) | **4.3710 (ppl 79.1)** |
| **mean ± std** | **4.6131 ± 0.0003** | **4.3881 ± 0.0009** | **4.3783 ± 0.0073** |

*Chance $= \ln(128) = 4.852$. All models learn.*  
*Train minutes per run: transformer ~3.8, mamba ~26.7, PCEI-tiny ~29.4.*  
*Forward throughput on CPU: transformer ~10–16k tok/s, mamba ~0.7–1.2k tok/s, PCEI-tiny ~0.8–0.9k tok/s. All runs finite, no NaNs.*

---

## 4. What the numbers say

* **SSM > attention on this task, decisively.** Both Mamba variants land at val loss ~4.38 vs the transformer's ~4.61, consistent across seeds (gap $\approx 0.22$ nats). Expected: the checksum subtask requires carrying a key across 7 fixed positions — recurrence is built for exactly that.
* **Ternary quantization does not hurt at matched params.** PCEI-tiny, with strict $\{-1, 0, +1\}$ weights in all 24 SSM projections, matches the dense Mamba control (4.378 vs 4.388 mean val loss). For the *"does ternary quantization destroy the model"* question, this is a clean no.
* **PCEI-tiny leads Mamba 2-for-2 on seeds** (both PCEI seeds below Mamba's best seed, 4.3872), mean gap 0.0098 nats in PCEI's favor. But: PCEI's own seed spread (0.0146) exceeds the gap, so noise cannot be ruled out with $n=2$. Honest claim: parity, leaning PCEI — not a proven win. A third seed would help resolve it.

---

## 5. What this proves and what it does not

**Proves:**
* The implementation is real and trains: ternary + FastKAN + Mamba compose, optimize cleanly, no pathologies.
* Aggressive ternary quantization is compatible with SSM language modeling at matched capacity — the central engineering bet of the PCEI text track survives a fair test.

**Does not prove:**
* Nothing about 750M scale. Small-scale results do not transfer automatically; scaling behavior is its own empirical question.
* Nothing about natural language. This is a toy grammar, not text.
* Not "PCEI > Mamba" (gap within noise) and not "PCEI > Transformer" in general (the task was constructed to reward recurrence).

**Epistemic status:** a screening test, and the architecture passed the screen. Screens like this kill most exotic architectures — had ternary collapsed the model, it would have shown as a clear loss here. Passing justifies spending the next round of compute; it does not conclude the research.

---

## 6. Limitations

1. **Tiny scale, synthetic data.** ~400k params, 600 steps, toy grammar. Nothing here predicts large-scale behavior.
2. **Task favors recurrence by construction.** The fixed 7-step key $\to$ checksum dependency plays to SSM strengths. A task rewarding in-context retrieval or copying could flip the transformer/SSM ranking.
3. **PCEI-tiny $\neq$ PCEINexus.** No diffusion steps, ambient field, neurogenesis, or response gating. Claims attach to the tested composition only.
4. **Two seeds cannot resolve the Mamba/PCEI gap.** 0.0098 nats against PCEI seed std 0.0073 — claim parity-leaning-PCEI, not victory.
5. **Throughput numbers are CPU-forward only**, using the repo's unoptimized pure-PyTorch scan (which materializes large `(B, L, d_inner, d_state)` temporaries). Do not generalize to GPU or to the $\mathcal{O}(1)$ recurrent inference path. The transformer's ~10k tok/s vs ~0.8k tok/s for the SSM variants on CPU is an implementation property, not an architectural verdict.
6. **No tuning.** Same LR/steps for all; nobody's best case is shown, only out-of-the-box behavior.

---

## 7. Setup findings (for the repo)

* `MambaConfig` silently defaults `d_inner=512` (not `d_model * expand`) unless `d_inner="auto"` is passed explicitly — a ~2x parameter blowup trap for anyone configuring small models.
* The repo's `_selective_scan_vectorized` cost is dominated by `(B, L, d_inner, d_state)` temporaries: 41 s/step at `d_inner=512` vs 3 s/step at `d_inner=192` in probing. Worth fusing before any serious training.

---

## 8. Conclusion

At matched parameter counts on a fixed synthetic task, a PCEI-style ternary + FastKAN SSM stack trains cleanly and matches-or-slightly-leads a dense Mamba, with both SSM variants decisively ahead of a transformer. The load-bearing result is the ternary one: strict 1.58-bit quantization did not degrade the model. Recommended next steps, in order of cost: (a) a third seed to resolve the Mamba/PCEI gap; (b) the same bake-off on a recurrence-neutral task (e.g., copying/retrieval-heavy); (c) a 10M-scale run on real text (TinyStories) before any 750M claims.

---

## 9. Reproducibility

* Scripts: `text_bakeoff.py` (experiment loop), `models_bakeoff.py` (the three model definitions)
* Results: `text_bakeoff_results.json`, Log: `text_bakeoff.log`
* Command: `cd ~/workspace/pcei-tests && ../pcei-venv/bin/python text_bakeoff.py`
* Requires the PCEI repository (for `MambaBlock`, `TernaryLinear`, `FastKANMixer`) and PyTorch. Deterministic given seeds; rerun to verify. ~2h wall-clock on 2 CPU cores for the full 6-run matrix.

*Note: the embodied-track bake-off (PCEIActorCritic vs MLP actor-critic, shared PPO loop, Pendulum-v1) was still running when this report was written and will be documented separately.*
