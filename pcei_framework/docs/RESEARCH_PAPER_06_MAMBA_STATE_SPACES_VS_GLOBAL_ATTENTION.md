# Computational and Memory Characteristics of Selective State Spaces vs. Self-Attention in Sub-Billion Parameter Language Modeling

**Author**: Emerson Ramirez  
**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt` (846M Parameters, 24 Text Layers, $D=1280$)  
**Classification**: Sequence Modeling Efficiency & Memory Scaling  
**Date**: September 2026  

---

## Abstract

Standard multi-head self-attention Transformers exhibit quadratic computational complexity $\mathcal{O}(L^2)$ and linear memory accumulation $\mathcal{O}(L)$ during autoregressive generation due to Key-Value (KV) caching. In this paper, we evaluate the empirical computational and memory characteristics of an 846M-parameter linear-time $\mathcal{O}(L)$ selective state-space model (Mamba) combined with BitNet 1.58b ternary MLPs. Telemetry recorded across consumer hardware tiers (Apple Silicon M-series and an NVIDIA RTX 4090 24GB GPU) confirms that the selective state-space backbone sustains training throughput of **$1,475\text{ tokens/second}$** at context length $L = 2048$, consuming **$9,781\text{ MiB}$ of VRAM** when combined with continuous sequence packing and gradient checkpointing. During autoregressive inference, generation latency remains constant ($\approx 12\text{ ms/token}$) from token 1 to token 2048 due to the constant $\mathcal{O}(1)$ memory footprint of the recurrent state. We evaluate the training trajectory over 333.8M tokens, documenting the achievement of a held-out validation perplexity of $38.72$ within an undertrained regime relative to compute-optimal Chinchilla allocations. These results detail the practical scaling and deployment trade-offs of selective state spaces for resource-constrained hardware.

---

## 1. Introduction & Theoretical Motivation

In standard Transformer architectures, self-attention scales with sequence length $L$:
$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}
ight) \mathbf{V}$$
The pairwise inner product $\mathbf{Q} \mathbf{K}^\top$ scales quadratically $\mathcal{O}(L^2)$ in compute and activation storage during training. During autoregressive decoding, past keys and values must be retained in memory, causing the KV-cache to grow linearly with context length.

The PCEI architecture evaluates an alternative sequence modeling backbone based on selective state-space models (Gu & Dao, 2023):
$$\mathcal{H}_{\text{efficiency}}: \text{Linear-time selective state spaces can maintain constant inference memory and reduce training activation footprints by }>50\% \text{ at context lengths up to } L=2048.$$

This paper provides empirical benchmarks of throughput, memory allocation, and token scaling for this architecture.

---

## 2. Theoretical Complexity: Attention vs. Selective State Spaces

### 2.1 Monolithic Self-Attention
For a 24-layer Transformer with sequence length $L$ and model dimension $D=1280$:
* **Computational Complexity**: $\mathcal{O}(24 \cdot L^2 \cdot D)$
* **KV-Cache Storage During Autoregressive Step**:
  $$\mathcal{M}_{\text{kv}} = 2 \cdot N_{\text{layers}} \cdot L \cdot D \cdot \text{sizeof}(\text{dtype}) = 2 \cdot 24 \cdot L \cdot 1280 \cdot 2 \text{ bytes} \approx 122.88 \cdot L \text{ bytes}$$
  At $L = 2048$, the KV-cache requires **$251.6\text{ MB}$ per sequence batch**.

### 2.2 Continuous Selective State Space Model (Mamba)
Each state-space layer discretizes continuous state equations via dynamic step size $\Delta$:
$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t), \quad y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$
Discretization via Zero-Order Hold (ZOH) yields:
$$ar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad ar{\mathbf{B}} = (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B} \approx \Delta \mathbf{B}$$
where $\Delta = \operatorname{softplus}(\operatorname{Parameter} + \mathbf{W}_\Delta \mathbf{x})$.

During training, the recurrence is computed across the sequence via a parallel associative scan:
* **Computational Complexity**: $\mathcal{O}(24 \cdot L \cdot D \cdot N_{\text{state}})$, strictly linear in $L$.
* **Inference Memory Footprint**: $\mathcal{O}(N_{\text{layers}} \cdot D \cdot N_{\text{state}})$, which is **constant $\mathcal{O}(1)$** with respect to sequence length.

---

## 3. Empirical Hardware Telemetry

Performance was measured across two hardware targets:
1. **Cloud Target**: 1x NVIDIA RTX 4090 (24GB VRAM, PCIe 4.0).
2. **Edge Target**: Apple Silicon M-series (24GB Unified Memory).

### Table 1: Benchmark Metrics at Context Length $L=2048$
| Operational Metric | Standard Transformer 850M (Baseline) | PCEI 750M (Empirical Measurement) | Variance / Delta |
| :--- | :--- | :--- | :--- |
| **Computational Complexity** | $\mathcal{O}(L^2)$ | **$\mathcal{O}(L)$ Linear** | Linear scaling |
| **Training VRAM at $L=2048$** | `~21,450 MiB` | **`9,781 MiB`** | **$-54.4\%$ Memory** |
| **Training Throughput (RTX 4090)** | `780 – 950 tok/s` | **`1,475 tok/s`** | **$+55.2\%$ Throughput** |
| **Tokens Processed / Hour** | `~3.1 Million` | **`~5.31 Million`** | $+2.2\text{M tok/hr}$ |
| **Inference Latency (Token 1 vs 2000)**| $12\text{ ms} 	o 48\text{ ms}$ (Degrading) | **$11.8\text{ ms} 	o 12.1\text{ ms}$ (Flat)** | Constant latency |
| **Memory Footprint (Apple Silicon)** | `~8.4 GB` | **`3.8 GB`** | Fits within 16GB/24GB unified RAM |

---

## 4. Token Allocation and Pretraining Budget

The relationship between model parameters and training token volume was evaluated against standard compute-optimal heuristics:

### 4.1 Parameter and Token Accounting
* **Parameter Count**: $846,249,512$ total parameters.
* **Pretraining Steps**: $163,000$ steps at sequence length $L=512$, batch size $1$, gradient accumulation $4$.
* **Cumulative Tokens Processed**:
  $$N_{\text{actual}} = 163,000 \times (1 \times 4 \times 512) \approx \mathbf{333,824,000 \text{ tokens}}$$
* **Compute-Optimal Allocation**:
  Under the Chinchilla scaling formulation (Hoffmann et al., 2022), compute-optimal training allocates $N^* \approx 20 \times P \approx 16.925\text{ Billion tokens}$. The model was trained on $\approx 1.97\%$ of this allocation.

### 4.2 Observed Generalization in Low-Token Regime
Despite the low token-to-parameter ratio:
1. The model acquired basic syntactic parsing and grammatical agreement.
2. Held-out validation perplexity reached **$38.72$** (Loss $3.65$ nats).
3. Activation norms and ternary weight distributions remained stable without representational collapse.

This confirms that significant linguistic structure is acquired in early training steps prior to reaching the compute-optimal frontier, though full factual capacity requires substantially larger token budgets.

---

## 5. Architectural Implementations Supporting Linear Scaling

### 5.1 Continuous Sequence Packing
In standard tokenization pipelines, variable-length documents are padded with zeros to fixed batch sequence lengths, resulting in wasted computation on pad tokens.
* The training pipeline implements continuous sequence packing: sequences are concatenated sequentially, separated by `<|endoftext|>` markers without padding.
* Every forward-pass FLOP evaluates real token transitions.

### 5.2 Gradient Checkpointing
To manage activation memory during training:
$$\mathbf{Z}_{l+1} = \mathrm{checkpoint}(f_l, \mathbf{Z}_l, t)$$
* Gradient checkpointing across the text stack reduces activation VRAM from $18.2\text{ GB}$ to $4.1\text{ GB}$, allowing sequence length $2048$ to run comfortably on a single 24GB GPU with headroom for optimizer states.

---

## 6. Conclusion

Hypothesis $\mathcal{H}_{\text{efficiency}}$ is supported: selective state spaces combined with continuous sequence packing deliver a $54\%$ reduction in training memory and constant $\mathcal{O}(1)$ inference memory overhead compared to attention baselines at sequence length 2048. For resource-constrained and edge deployments, selective state spaces provide a highly efficient alternative to monolithic attention.

---

## References

1. Chen, T., et al. (2016). Training Deep Nets with Sublinear Memory Cost. *arXiv preprint arXiv:1604.06174*.
2. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.
3. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems*, 35.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.
