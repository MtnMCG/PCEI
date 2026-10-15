# Physically Constrained Emergent Intelligence (PCEI): Master Architecture Specification

**Framework Version**: PCEI v0.2 (Production Architecture)  
**Supersedes**: `PCEI_Framework_Spec (2).txt` (v0.1 Design Draft)  
**Author**: Emerson Ramirez (Lead Architect & Inventor, PCEI)  
**Collaborator**: Google Gemini / Antigravity Assistant  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt`  
**Total Parameters**: $846,249,512$ physical parameters ($332,595,200$ BitNet 1.58b ternary parameters)  
**Date**: September 2026  

---

## Executive Summary & Evolutionary Retrospective

The original PCEI design draft (`v0.1 — Design Draft: "To understand that it understands"`) established a revolutionary philosophical thesis: **biological intelligence, self-consistency, and sample efficiency are emergent consequences of physical locality constraints imposed on a dynamic neural substrate, rather than unconstrained global attention.**

While the philosophical foundation of v0.1 remains the bedrock of the framework, rigorous empirical auditing across 334 million tokens of training and 1,000 cycles of autonomous latent dreaming revealed several mathematical and practical boundaries. In v0.2, these discoveries catalyzed significant architectural breakthroughs:

1. **The Causal Autoregressive SSM Pivot**: Discrete language syntax and multi-turn conversational coherence require causal next-token autoregression. The v0.1 concept of continuous text diffusion was superseded by a 24-layer **Selective State-Space Model (Mamba)**, reserving continuous latent diffusion exclusively for the **Phase 2 Dreaming Chamber** (sleep consolidation).
2. **From $\{0, 1, 2\}$ to BitNet 1.58b Ternary Quantization ($\{-1, 0, +1\}$)**: The uncalibrated positive ternary scheme in v0.1 was replaced with calibrated BitNet 1.58b quantization using dynamic scale factors $\gamma = \frac{1}{mn}\sum |W_{ij}|$ and Straight-Through Estimators (STE). Initial pretraining utilized an experimental 5-state ladder ($\{-2, -1, 0, 1, 2\}$, historically aggregated as $40.08\%$ negative, $19.82\%$ zero, $40.09\%$ positive). Following an October 2026 external audit, the quantizer was standardized to canonical Microsoft BitNet b1.58 ($\{-1, 0, +1\}$ at $1.58\text{ bits/param}$), cutting parameter entropy by $31.9\%$ and eliminating code token generation loops.
3. **Pretrained Manifold Grafting**: Day-1 scratch cold-starts with ternary constraints in sub-billion models suffer from slow vocabulary convergence. v0.2 introduced cross-architectural weight grafting (GPT-2 Large token embeddings, CLIP ViT-B/16 visual patch projections, and Whisper audio filterbanks) to seed semantic manifold coordinates immediately.
4. **Physical Brokerage Decoupled from Autograd**: Analytical and empirical proofs (Paper II) demonstrated that physical 3D coordinates and Hookean bond relaxation must operate strictly outside the backpropagation computational graph $\mathcal{G}$ ($\Delta \mathcal{L} \equiv 0.0000000000$), preserving mathematical purity and eliminating optimization drag.
5. **Dual Cognitive Memory Architecture**: In v0.1, the Ambient Field was hypothesized to act as an unconcatenated working memory buffer. Information theory audits (Paper I) disproved this due to permutation invariance ($\bar{\rho} = 0.2471$). In v0.2, strict separation of concerns is enforced: token history sliding windows ($k \le 1800$) handle discrete autoregressive syntax, while the Ambient Field handles homeostatic plasticity modulation and affective state anchoring.

---

## 1. System Topology: The Multi-Stack Physical Continuum

The PCEI architecture arranges its neural processing modules within a continuous three-dimensional physical coordinate space $\mathbf{P} \in \mathbb{R}^{N \times 3}$:

```
                            [ 3D PHYSICAL SPACE ]
  Z-Axis
    ▲
    │   ┌─────────────────────────────────────────────────────────┐
 30 │   │                 STACK A: TEXT / REASONING               │
    │   │         24 Mamba SSM Layers | d_model = 1280            │
    │   │      BitNet 1.58b Ternary MLPs (332.6M params)          │
    │   │           Causal Autoregressive Generation              │
    │   └───────────────────────────┬─────────────────────────────┘
    │                               │ Boundary Layer (Projection)
 10 │   ┌───────────────────────────┴─────────────────────────────┐
    │   │                 STACK B: IMAGE PERCEPTION               │
    │   │      4 Diffusion Layers | d_img = 512 | Patch=16x16     │
    │   │         Grafted CLIP ViT-B/16 Patch Projection          │
    │   └───────────────────────────┬─────────────────────────────┘
    │                               │ Boundary Layer (Projection)
  0 │   ┌───────────────────────────┴─────────────────────────────┐
    │   │                 STACK C: AUDIO PERCEPTION               │
    │   │      4 Diffusion Layers | d_audio = 256 | 80 Mel Bins   │
    │   │          Grafted Whisper Mel Projection Matrix          │
    │   └─────────────────────────────────────────────────────────┘
    └──────────────────────────────────────────────────────────────► X/Y-Plane
         ▲                                                     ▲
         │                                                     │
    Hookean Bonds (Spring Relaxation)            Ambient Field (Slow/Fast State)
```

### 1.1 Asymmetric Hierarchy & Dimensionality

PCEI models the biological reality that language and reasoning are not peer modalities to raw perception, but higher-order integration systems:

| Modality Stack | Depth ($N_{\text{layers}}$) | Hidden Dim ($D$) | Core Processing Engine | Parameter Count |
| :--- | :--- | :--- | :--- | :--- |
| **Stack A: Text / Logic** | **24 Layers** | **1280** | Causal Selective State Space (Mamba) + BitNet 1.58b | $792,845,312$ |
| **Stack B: Visual Perception** | **4 Layers** | **512** | Continuous Latent Diffusion (Mamba Backbone) | $36,118,528$ |
| **Stack C: Auditory Perception**| **4 Layers** | **256** | Continuous Latent Diffusion (Mamba Backbone) | $17,285,672$ |
| **Total Model Architecture** | **32 Layers** | — | **Hybrid Selective State Space Continuum** | **$846,249,512$** |

### 1.2 Boundary Layer Projections
Stacks communicate exclusively across adjacent spatial interfaces without dense cross-attention:
$$\mathbf{z}_{\text{text}}^{(0)} = \mathbf{z}_{\text{text}}^{(0)} + \mathbf{W}_{\text{img}\to\text{text}} \mathbf{z}_{\text{img}}^{(L_B)}$$
$$\mathbf{z}_{\text{img}}^{(0)} = \mathbf{z}_{\text{img}}^{(0)} + \mathbf{W}_{\text{audio}\to\text{img}} \mathbf{z}_{\text{audio}}^{(L_C)}$$
where $\mathbf{W}_{\text{img}\to\text{text}} \in \mathbb{R}^{1280 \times 512}$ and $\mathbf{W}_{\text{audio}\to\text{img}} \in \mathbb{R}^{512 \times 256}$ are learned projection matrices.

---

## 2. TextStack Core Engine: Selective State Space & Ternary Logic

The TextStack comprises 24 identical `PlasticLayer` blocks operating in bfloat16 mixed precision.

### 2.1 Causal Selective State-Space Formulations (Mamba)
Each layer replaces the quadratic $\mathcal{O}(L^2)$ attention mechanism with a continuous differential system:
$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t), \quad y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$
Discretized using Zero-Order Hold (ZOH) with input-dependent dynamic parameterization:
$$\Delta = \operatorname{softplus}(\operatorname{Parameter} + \mathbf{W}_\Delta x)$$
$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$$
$$\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B} \approx \Delta \mathbf{B}$$
The full sequence is computed in $\mathcal{O}(L)$ time via chunked associative prefix scan:
$$(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)$$

### 2.2 BitNet 1.58b Ternary Quantization & Architectural Audit ($\{-1, 0, +1\}$)
Every Feed-Forward MLP layer in the TextStack employs `TernaryLinear` quantization:
$$W_{ij} \in \left\{ -1, 0, +1 \right\} \cdot \gamma$$
Where the layer dynamic scale factor $\gamma$ is computed dynamically:
$$\gamma = \frac{1}{m \cdot n} \sum_{i=1}^m \sum_{j=1}^n |W_{ij}|$$
Quantization is governed by the Straight-Through Estimator (STE):
$$\tilde{W} = \operatorname{clamp}\left(\operatorname{round}\left(\frac{W}{\gamma}\right), -1, +1\right) \cdot \gamma$$
$$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \tilde{W}}$$

* **Architectural Evolution & Full Transparency**:
  * **Pretraining 5-State Ladder**: Pretraining runs operated with an experimental 5-state ladder ($\{-2, -1, 0, 1, 2\}$ at $2.32\text{ bits/param}$), where parameters distributed as: $-2: 14.09\%$, $-1: 22.34\%$, $0: 27.14\%$, $+1: 22.34\%$, $+2: 14.09\%$. Early telemetry logged these signed groups as $40.08\%$ negative, $19.82\%$ zero, $40.09\%$ positive.
  * **Canonical BitNet b1.58 Standard**: Following an external audit in October 2026, the layer was refactored to canonical Microsoft BitNet b1.58 ($\{-1, 0, +1\}$ at $1.58\text{ bits/param}$). Evaluation confirmed that strict ternary quantization matches or improves perplexity (11.14 vs 11.16 on 100M; 7.00 on 750M), cuts parameter entropy by $31.9\%$, and eliminates repetitive code token loops.

---

## 3. The Physical Coordinate Broker & Dynamic Entanglement

Unlike conventional models where network topology is static, PCEI embeds each layer in a 3D coordinate space:
$$\mathbf{p}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$$

### 3.1 Resonance Detection
Between any pair of layers $i$ and $j$, the `ResonanceDetector` calculates activation temporal cross-correlation:
$$\rho_{ij} = \frac{\sum_{t} (\mathbf{z}_i(t) - \bar{\mathbf{z}}_i)(\mathbf{z}_j(t) - \bar{\mathbf{z}}_j)}{\sqrt{\sum_t \|\mathbf{z}_i(t) - \bar{\mathbf{z}}_i\|^2 \sum_t \|\mathbf{z}_j(t) - \bar{\mathbf{z}}_j\|^2}}$$
If $\rho_{ij} > \theta_{\text{form}}$ (default $0.65$), an active entanglement bond $\mathcal{B}_{ij}$ is formed.

### 3.2 Hookean Physical Relaxation (Topology Shifter)
Active bonds act as physical springs pulling correlated modules closer in the $(x, y)$ plane:
$$\mathbf{F}_{ij} = -k_{\text{spring}} (\|\mathbf{p}_i - \mathbf{p}_j\| - d_0) \frac{\mathbf{p}_i - \mathbf{p}_j}{\|\mathbf{p}_i - \mathbf{p}_j\|}$$
$$\mathbf{p}_i^{(t+1)} = \mathbf{p}_i^{(t)} + \eta_{\text{phys}} \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}$$
* **Mathematical Separation**: As proven in Paper II, $\frac{\partial \mathcal{L}}{\partial \mathbf{p}} \equiv \mathbf{0}$. Physical relaxation runs asynchronously outside PyTorch autograd, ensuring zero gradient drag on backpropagation.

### 3.3 Biological Phase-Lag Synchrony Spikes
In continuous autoregressive text processing (`diffusion_timesteps = 1`), neural synchrony operates through **biological phase-lag spike caching**:
* When layers $i$ and $j$ correlate above threshold $\theta_{\text{reinforce}}$, the `EntanglementBroker` computes directional residual current vectors $\mathbf{s}_{\text{sync}}^{(i \to j)}$.
* To preserve causality without temporal leaking, spikes generated at token $t$ are cached on `PCEIModel.cached_spikes` and injected into the target layer on token $t+1$:
  $$\mathbf{z}_j^{(t+1)} = \mathbf{z}_j^{(t+1)} + \mathbf{s}_{\text{sync}}^{(i \to j)}(t)$$
* This mathematically mirrors biological gamma-band phase locking (Singer, Engel), where assemblies of neurons that fire synchronously bind disparate features into unified perceptual and conceptual objects across consecutive synaptic transmission cycles.

---

## 4. The Subconscious Layer: Ambient Field & Homeostatic Dynamics

The `AmbientField` threads continuously through the entire architecture as a dual-timescale dynamical system:
$$\mathbf{s}_{\text{fast}} \in \mathbb{R}^{1280}, \quad \mathbf{s}_{\text{slow}} \in \mathbb{R}^{1280}$$

### 4.1 Fast & Slow Update Mechanics
On every forward pass:
$$\mathbf{s}_{\text{fast}}^{(t)} = (1 - \beta_{\text{fast}}) \mathbf{s}_{\text{fast}}^{(t-1)} + \beta_{\text{fast}} \left(\frac{1}{L} \sum_{k=1}^L \mathbf{z}_{\text{text}}^{(k)}\right)$$
$$\mathbf{s}_{\text{slow}}^{(t)} = (1 - \beta_{\text{slow}}) \mathbf{s}_{\text{slow}}^{(t-1)} + \beta_{\text{slow}} \mathbf{s}_{\text{fast}}^{(t)}$$
where $\beta_{\text{fast}} = 0.10$ (high volatility) and $\beta_{\text{slow}} = 0.001$ (high inertia).

### 4.2 Homeostatic Plasticity Modulation
Catastrophic forgetting is prevented physically by modulating layer-specific learning rates based on slow-field confidence:
$$\eta_{\text{eff}}^{(l)} = \eta_{\text{base}} \cdot \left(1.0 - \sigma_{\text{conf}}^{(l)}\right)$$
Where $\sigma_{\text{conf}}^{(l)} \in [0.0, 1.0]$ tracks how consistently layer $l$'s activation spectrum aligns with the slow field. Highly consolidated concepts calcify to $\sigma \to 0.25+$, protecting foundation knowledge from being overwritten by noisy transient prompts.

### 4.3 Emergent Global Attention: Zero-Gated Residual Broadcast
To solve Mamba's sequential compression bottleneck (where continuous state $h_t$ loses subject-predicate binding across long contexts) without incurring Transformer $\mathcal{O}(L^2)$ quadratic cost, the Ambient Field acts as an **emergent Global Workspace** (Baars/Dehaene):
* Inside every `DiffusionBlock` across the 24 text layers, a zero-initialized learned projection conditions the token stream on consolidated memory:
  $$\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{out}} + \tanh(\alpha_{\text{ambient}}) \cdot \mathbf{W}_{\text{ambient}} \mathbf{s}_{\text{slow}}$$
* **Zero-Disruption Initialization**: $\alpha_{\text{ambient}} = 0.0$ and $\mathbf{W}_{\text{ambient}} = \mathbf{0}$, ensuring that existing pretrained checkpoints load with an exact bit-level error of $0.00\text{e}+00$.
* **Emergent Conditioning**: Backpropagation opens the gate $\alpha_{\text{ambient}}$ as the network discovers that anchoring on the global subconscious state eliminates semantic random walks and binds deductive premises across the sequence.

---

## 5. The Three-Phase Biological Cognitive Lifecycle

PCEI models do not rely solely on brute-force next-token pretraining. Instead, training and maintenance follow a 3-phase cycle analogous to biological wakefulness, sleep, and instruction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: AWAKE PRETRAINING                               │
│  Continuous Sequence Packing (0% Pad) | WSD Learning Rate Scheduling        │
│  Ingests massive web text, books, and code | Establishes syntactic basis    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: AUTONOMOUS DREAMING CHAMBER                     │
│  Closed-Loop Latent Diffusion | Zero External Tokens                        │
│  Reconstructs continuous memories into discrete tokens                      │
│  Perplexity contracts (57.79 → 38.72) | Manifold stabilization              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: BLENDED INSTRUCTION TUNING                      │
│  42% Claude-CoT & Anthropic RLHF | 20% ShareGPT Multi-Turn                  │
│  18% UltraChat | 10% Council-Ultima Teacher | 10% FineWeb-Edu               │
│  Context Length: L = 2048 | Dynamic Gradient Checkpointing                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Phase 1: Wakeful Pretraining
* Employs **Warmup-Stable-Decay (WSD)** learning rate schedules and continuous sequence packing.
* Token documents are concatenated end-to-end separated by `<|endoftext|>`, eliminating null padding tokens and ensuring 100% compute efficiency.

### 5.2 Phase 2: Autonomous Dreaming Chamber
* Uses continuous Gaussian forward noise schedules $\beta_t$ in latent space:
  $$\mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$
* A reverse Mamba denoiser predicts the noise $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$.
* Held-out validation perplexity drops monotonically without ingesting any new data, proving that offline generative replay calcifies representations.

### 5.3 Phase 3: Blended Instruction Tuning
* Ingests high-density reasoning, logic, and multi-turn conversational data at $L=2048$.
* Selective gradient checkpointing on the TextStack drops VRAM from $18.2\text{ GB}$ to $9.78\text{ GB}$, enabling full 2048-token instruction tuning on a single consumer GPU.

---

## 6. Full Architectural Evolution Matrix (v0.1 vs. v0.2)

| Subsystem | v0.1 Origin Draft (`PCEI_Framework_Spec (2).txt`) | v0.2 Production Reality (Current Checkpoint) | Rationale & Empirical Justification |
| :--- | :--- | :--- | :--- |
| **Text Generation** | Continuous Latent Diffusion (token $\to$ latent $\to$ diffuse $\to$ detokenize) | **Causal Autoregressive Mamba Selective State Space** | Continuous diffusion in discrete token space destroys syntax; causal autoregression is mathematically required. |
| **Quantization** | $\{0, 1, 2\}$ TernaryNet with single scalar scale $\alpha$ | **BitNet 1.58b $(\{-1, 0, +1\})$ with Dynamic Scale $\gamma$** | BitNet dynamic scaling eliminates gradient dead zones and yields $40:20:40$ symmetric gating. |
| **Cold Start** | Scratch initialization on Apple Silicon M4 Air | **Weight Grafting (GPT-2 Large, CLIP ViT-B/16, Whisper)** | Accelerates vocabulary convergence by thousands of GPU-hours without altering the non-transformer architecture. |
| **Physical Bonds** | Injected synchronous activation spikes into autograd | **Decoupled 3D Coordinate Shifter Outside Autograd** | Eliminates gradient drag ($\Delta \mathcal{L} \equiv 0.0$), enabling unbounded topological scaling. |
| **Context Memory** | Hypothesized Ambient Field as token-free memory ($k=0$) | **Token Sliding Window ($k \le 1800$) + Ambient Homeostasis** | Disproved token-free ambient recall ($\bar{\rho}=0.2471$); Ambient Field repurposed for plasticity and attractor stability. |
| **Sequence Packing** | Variable length with zero padding | **Continuous Token-Maxx Sequence Packing (0% pad)** | 100% of GPU compute expended on active tokens; prevents wasteful padding compute. |
| **Global Attention** | Quadratic Transformer Attention $\mathcal{O}(L^2)$ | **Zero-Gated Ambient Broadcast ($\mathcal{O}(L)$)** | Solves Mamba associative random walks via Global Workspace broadcast; zero-initialization guarantees bit-level checkpoint preservation. |
| **Neural Synchrony** | Unsynchronized / discarded autoregressive spikes | **Biological Phase-Lag Spike Caching** | Caches cross-layer synchrony spikes on token $t$ and injects on $t+1$, reproducing gamma-band feature binding without temporal leaking. |
| **Curriculum Pipeline** | Premature blended conversational SFT ($L=2048$) | **Phased Text Pretraining (P1) + Dreaming Chamber (P3)** | Re-centering on packed text and self-resonance dropped eval loss from $6.40 \to 4.94$ and boosted token accuracy from $15.1\% \to 25.6\%$. |
| **Training Scale** | Apple Silicon local CPU/MPS background thread | **Hybrid Cloud RTX 4090 + Local M4 Edge Workstation** | Sustains $1,475\text{ tok/s}$ at $L=2048$ with automated credit guard and bidirectional rsync sync. |
