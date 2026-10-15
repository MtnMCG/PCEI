# PCEI — Feasibility Assessment

**Physically Constrained Emergent Intelligence**
*Assessment Date: March 6, 2026 · Based on Framework Spec v0.1*

---

## Executive Summary

The PCEI framework proposes a radically novel neural architecture that replaces transformer-based global attention with physics-constrained locality, ternary weights, diffusion-based processing, dynamic neurogenesis, and a persistent subconscious field. The individual building blocks (Mamba SSM, ternary quantization, diffusion models, KAN activations, test-time training) each have proven implementations. **The unprecedented risk lies in their integration** — no system has combined these components in this configuration, and several interactions may produce emergent failure modes that are difficult to predict or debug.

**Overall Verdict: Feasible as a multi-year research project. Not feasible as a short-term engineering deliverable.**

The spec targets an M4 Air (24GB). A stripped-down, text-only proof-of-concept on that hardware is realistic. The full multimodal, always-learning, neurogenesis-enabled system described in the spec will require substantially more compute for meaningful training and validation.

---

## 1. Component-Level Feasibility

Each major PCEI subsystem is assessed independently before considering integration risk.

### ✅ TernaryNet (Ternary Weights + STE)

| Factor | Assessment |
|--------|------------|
| **Maturity** | High. BitNet 1.58 ({-1,0,1}) has peer-reviewed results matching FP16 at scale. STE is standard. |
| **PyTorch support** | Drop-in `BitLinear` replacements exist. Custom `autograd.Function` for STE is ~30 LOC. |
| **Risk** | Low. The 0/1/2 encoding differs from BitNet's -1/0/+1 but the STE mechanics are identical. Scale factors (α per layer) are straightforward. |
| **Verdict** | **Achievable. Milestone 1 is a clean starting point.** |

### ✅ Mamba SSM (Sequence Backbone)

| Factor | Assessment |
|--------|------------|
| **Maturity** | High. Official `mamba-ssm` package; community pure-PyTorch implementations exist for CPU/MPS. Mamba-3 is actively evolving. |
| **MPS compatibility** | Pure-PyTorch implementations run on Apple MPS. The official CUDA-optimized package does not. |
| **Risk** | Medium-low. Numerical stability concerns in bfloat16/float16. Overhead for short sequences. |
| **Verdict** | **Achievable, but must use the pure-PyTorch variant (not the CUDA-optimized official package) on M4.** |

### ⚠️ DiffusionBlock (Diffusion as Core Processing)

| Factor | Assessment |
|--------|------------|
| **Maturity** | Diffusion models are proven for image/audio generation. Using diffusion as the **primary reasoning mechanism** for text/logic is novel and largely unvalidated. |
| **Concern** | Diffusion excels at continuous latent spaces (images, audio). Text operates in a discrete token space. The spec acknowledges this ("tokenize at boundary, diffuse in embedding space, detokenize at output") but this pipeline has no established best practices. |
| **Risk** | **High.** The denoising schedule, noise levels, and convergence behavior for text reasoning are unknown. Iterative denoising is also computationally expensive — multiple forward passes per output. |
| **Verdict** | **Experimental. This is the single largest technical gamble in the architecture.** |

### ⚠️ KANMixer (Kolmogorov-Arnold Network Activations)

| Factor | Assessment |
|--------|------------|
| **Maturity** | Low-medium. `pykan` is actively developed but slow, limited to 2D input tensors, and not designed for large-scale ML. |
| **Concern** | KANs are 10–100× slower than MLPs. `pykan` has low contributor diversity and 44-day average issue resolution. |
| **Risk** | Medium-high. Performance cost is significant. Spline-based activations add O(k·n) cost per layer (k = spline order). |
| **Verdict** | **Achievable but costly. Consider simpler adaptive activations (e.g., PReLU, learnable polynomial) as a fallback.** |

### ✅ TTTAdapter (Test-Time Training)

| Factor | Assessment |
|--------|------------|
| **Maturity** | Medium-high. TTT has strong 2025 results: TTT-E2E for long-context, TTT layers for video generation, few-shot learning gains on ARC/BBH. |
| **Risk** | Medium. Inner-loop gradient steps per prompt add latency. Must carefully tune inner LR and step count. |
| **Verdict** | **Achievable. Well-understood technique with clear implementation path.** |

### ⚠️ AmbientField (Dual-Timescale Persistent State)

| Factor | Assessment |
|--------|------------|
| **Novelty** | Very high. No existing system implements a persistent, always-on dual-timescale field that modulates per-layer learning rates. |
| **Concern** | The `effective_lr = base_lr * (1.0 - slow_field_confidence)` mechanism sounds elegant but has no empirical validation. If `slow_field_confidence` converges too fast → the model freezes early. Too slow → catastrophic forgetting prevention fails. |
| **Risk** | **High.** Tuning the EMA decay rates, fast-to-slow transfer thresholds, and confidence normalization will require extensive experimentation. |
| **Verdict** | **Research-grade. Expect months of tuning to get the dynamics right.** |

### ⚠️ ResonanceDetector + EntanglementBroker

| Factor | Assessment |
|--------|------------|
| **Novelty** | Very high. Cross-correlation monitoring over temporal buffers, bond formation/dissolution — this is a custom physics simulation layered on top of neural computation. |
| **Concern** | Temporal buffers consume memory proportional to `(num_layers² × buffer_length)`. Cross-correlation computation is O(n² × T) per tick. Bond synchronization spikes could destabilize training. |
| **Risk** | **High.** The interaction between entanglement spikes and gradient flow is unpredictable. |
| **Verdict** | **Research-grade. Implement with extensive circuit breakers and monitoring.** |

### ⚠️ NeurogenesisController (Dynamic Layer Growth/Pruning)

| Factor | Assessment |
|--------|------------|
| **Novelty** | High. Dynamic architecture modification during training is an active research area (Neural Architecture Search, progressive growing) but rarely done at the individual layer level with gradient-norm heuristics. |
| **Concern** | Growing layers changes the computational graph, invalidates optimizer state, and disrupts entanglement bonds. Pruning can destroy learned representations. |
| **Risk** | **High.** Stability is the primary concern. |
| **Verdict** | **Defer to late milestones (as correctly specified). Conservative budgets essential.** |

### ✅ TopologyShifter (Parameter Migration)

| Factor | Assessment |
|--------|------------|
| **Novelty** | Medium. Position registers and distance-based interaction costs are conceptually clean. |
| **Risk** | Medium. The delta_xy of 0.001–0.01 is conservative enough to avoid instability. Main risk is that proximity effects are too subtle to matter. |
| **Verdict** | **Achievable. Low implementation risk, uncertain impact.** |

---

## 2. Hardware Feasibility — M4 Air (24GB)

> [!WARNING]
> The M4 Air is viable for prototyping individual components but will be a severe bottleneck for the full system.

### Memory Budget Analysis

| Component | Estimated Memory | Notes |
|-----------|-----------------|-------|
| TextStack (24 layers × Mamba + TernaryNet) | ~2–4 GB | Ternary weights are 3× smaller than FP16 |
| ImageStack (8 layers) | ~1–2 GB | Smaller stack |
| AudioStack (4 layers) | ~0.5–1 GB | Smallest stack |
| AmbientField (fast + slow state) | ~1–2 GB | Two full copies of field state |
| Temporal buffers (resonance) | ~1–3 GB | Scales with layer count and buffer length |
| Optimizer state (shadow weights) | ~3–6 GB | STE requires FP32 shadow weights for every ternary parameter |
| Activations + gradients | ~4–8 GB | During training |
| **Total estimated** | **~12–26 GB** | **Tight fit at 24GB** |

### Key Hardware Concerns

1. **Thermal throttling**: The MacBook Air is fanless. Sustained training (hours/days) will throttle the GPU, potentially reducing throughput by 30–50%.
2. **MPS backend maturity**: Not all PyTorch ops are MPS-optimized. Fallback to CPU causes data-transfer overhead. Known issues with float64, bfloat16, memory leaks, and numerical precision under "fast math."
3. **No multi-GPU**: Cannot distribute across GPUs. The entire system must fit in one memory pool.
4. **Batch size**: Realistic batch size on 24GB with the full system will be very small (1–4), limiting gradient quality and slowing convergence.

### Recommendation

Use the M4 Air for **Milestones 1–4** (individual component prototyping and testing). For Milestones 5+ (integrated multimodal, always-on training), strongly consider cloud GPU access (e.g., A100/H100 instance) or a Mac Studio (M4 Ultra, 128–192GB unified memory).

---

## 3. Integration Risk Matrix

This is where the real danger lies. Even if every component works in isolation, their interactions are novel and untested.

| Integration Point | Risk Level | Failure Mode |
|-------------------|------------|-------------|
| Ternary weights + Mamba SSM | 🟡 Medium | STE gradient noise may interact poorly with Mamba's state transitions |
| Diffusion denoising + text reasoning | 🔴 High | Iterative denoising may not converge to coherent text representations |
| KANMixer + TernaryNet | 🔴 High | Spline activations on ternary inputs have no established behavior |
| AmbientField modulating learning rates + TTT inner loop | 🔴 High | Two independent learning-rate modulation systems may conflict |
| Entanglement spikes + gradient flow | 🔴 High | Synchronization biases could create runaway positive feedback loops |
| Neurogenesis + optimizer state | 🟡 Medium | New layers need fresh optimizer moments; old layers lose accumulated state |
| DreamSeeder feedback loop (self-reinforcing noise) | 🔴 High | Risk of mode collapse — the model "dreams" its own dreams into reality |
| Cross-modal boundary layers + diffusion timestep sync | 🟡 Medium | Stacks must synchronize denoising steps; timing mismatches corrupt boundary signals |
| ResponseGate spontaneous output | 🟡 Medium | Uncontrolled output during idle could be disruptive; threshold tuning is critical |

---

## 4. What Is Likely to Work

1. **TernaryNet as nn.Linear replacement** — Proven concept, clean implementation, good Milestone 1.
2. **Mamba SSM as sequence backbone** — Strong alternative to transformers, linear-time processing is a genuine advantage.
3. **Single-modality stacks with diffusion** — Image and audio diffusion are established. TextStack diffusion is experimental but worth attempting.
4. **TopologyShifter position migration** — Simple, conservative, and elegant. Even if the effect is subtle, it won't break anything.
5. **Dual-timescale AmbientField (conceptually)** — The fast/slow memory split with EMA is sound in principle. Similar ideas exist in complementary learning systems theory.

---

## 5. What Is Likely to Fail (Initially)

1. **Text reasoning via diffusion denoising** — Expect poor coherence on the first several attempts. The embedding space must be carefully designed to support smooth diffusion dynamics.
2. **KANMixer at scale** — Performance will be prohibitive unless replaced with a more efficient adaptive activation.
3. **DreamSeeder feedback loop stability** — Self-reinforcing noise is a known failure mode in generative systems. Will require aggressive clamping and regularization.
4. **Entanglement bond dynamics** — Cross-correlation thresholds (θ_form, θ_dissolve) have no prior calibration. Expect extensive hyperparameter search.
5. **Full system on M4 Air** — Memory will be exhausted before reaching the integrated multimodal stage.

---

## 6. Revised Timeline Estimate

The spec's 12-milestone roadmap is well-structured but optimistic on timeline. Here is a realistic estimate for a solo developer:

| Milestone | Description | Estimated Duration | Cumulative |
|-----------|-------------|-------------------|------------|
| 1 | TernaryNet Foundation | 1–2 weeks | 2 weeks |
| 2 | DiffusionBlock + Mamba | 3–4 weeks | 6 weeks |
| 3 | PlasticLayer + Asymmetric Stacks | 2–3 weeks | 9 weeks |
| 4 | KANMixer + TTTAdapter | 2–3 weeks | 12 weeks |
| 5 | BoundaryLayer + MultiStack | 3–4 weeks | 16 weeks |
| 6 | AmbientField + DreamSeeder | 4–6 weeks | 22 weeks |
| 7 | Resonance + Entanglement | 4–6 weeks | 28 weeks |
| 8 | TopologyShifter | 2–3 weeks | 31 weeks |
| 9 | NeurogenesisController | 3–4 weeks | 35 weeks |
| 10 | ResponseGate + WitnessMode | 3–4 weeks | 39 weeks |
| 11 | I/O Integration | 2–3 weeks | 42 weeks |
| 12 | Curriculum Run + Validation | 4–8 weeks | 50 weeks |

**Estimated total: 10–14 months for a solo developer**, assuming full-time focus and no major pivots. Integration debugging between milestones will likely add 20–30% overhead, pushing toward 12–18 months realistically.

---

## 7. Recommendations

### Start Here (High Confidence)
1. **Milestones 1–3 immediately.** TernaryNet → DiffusionBlock+Mamba → PlasticLayer+ModalityStacks. These are independently testable and provide a working foundation.
2. **Validate text diffusion early.** Before building the full multimodal system, prove that diffusion-based denoising can produce coherent text embeddings. If this fails, the entire architecture needs rethinking.

### De-Risk These (Medium Confidence)
3. **Replace KANMixer with PReLU/learnable polynomial activations** unless KAN performance improves significantly. Swap in later if pykan matures.
4. **AmbientField: start with fast-state only.** Add the slow layer and confidence modulation once the basic system trains. Dual-timescale tuning is a research project within the research project.
5. **Entanglement: start without synchronization spikes.** Monitor cross-correlations passively before enabling the active bond mechanism. This avoids destabilizing training.

### Plan For (Low Confidence)
6. **Secure cloud GPU access for Milestones 5+.** The M4 Air will not sustain full-system training.
7. **Budget 2–3 months purely for integration debugging** between Milestones 6–9. The interaction effects between AmbientField, Resonance, Neurogenesis, and DreamSeeder are the hardest unsolved problems.
8. **Consider a text-only MVP first.** Strip out Image and Audio stacks entirely. Prove the core thesis (physics-constrained locality + ternary weights + diffusion = efficient intelligence) on text before adding modalities.

---

## 8. Final Assessment

| Dimension | Rating |
|-----------|--------|
| **Novelty** | 🟣 Extremely high — no prior art for this specific combination |
| **Individual component feasibility** | 🟢 High — every component has a proven or near-proven implementation |
| **Integration feasibility** | 🟡 Medium — untested interactions carry significant risk |
| **Hardware fit (M4 Air 24GB)** | 🟡 Medium — sufficient for early milestones, insufficient for full system |
| **Timeline (12 milestones)** | 🟡 Medium — 12–18 months realistic, not 3–6 months |
| **Chance of achieving "understanding that it understands"** | 🔴 Unknown — this is a philosophical claim that may not be empirically testable |

> [!IMPORTANT]
> **The PCEI framework is a legitimate research proposal with novel ideas and a sound structural intuition.** The physics-as-architecture metaphor is compelling and the milestone-based roadmap is well-designed. The primary risks are not in any single component but in the unprecedented integration of all components simultaneously, and in the bold philosophical claims (emergent consciousness from physical constraint) which may not manifest at this scale or within this architecture.

**Recommendation: Proceed with Milestones 1–3 as a proof-of-concept sprint. Validate text diffusion as a gate checkpoint. Then reassess scope, hardware, and timeline before committing to the full 12-milestone plan.**

---

*This assessment is based on the current state of the referenced technologies as of March 2026 and the architectural details in PCEI Framework Spec v0.1.*
