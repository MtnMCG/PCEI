# Weight Decay Dynamics, Discrete Quantization, and Empirical Generalization in Low-Token Regimes

**Author**: Emerson Ramirez  
**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt` (846M Parameters, 24 Text Layers, $D=1280$)  
**Classification**: Learning Theory, Quantization Dynamics & Parameter Capacity  
**Date**: September 2026  

---

## Abstract

Standard large language models operate in continuous 16-bit floating-point parameter spaces where parameter capacity can be utilized for memorizing factual lookup associations. In this paper, we analyze the effect of strong architectural capacity constraints—specifically continuous metabolic weight decay, discrete BitNet 1.58b ternary quantization, and dynamic bond dissolution—on the learning dynamics of an 846M-parameter selective state-space model trained on a constrained budget of $333.8\text{ million tokens}$. We formalize the capacity boundary: limiting weights to $\{-1, 0, +1\} \cdot \gamma$ restricts parameter entropy to $\le \log_2(3) \approx 1.58\text{ bits/parameter}$, while continuous weight decay drives low-frequency gradient pathways into a zero-valued sparse state ($19.82\%$ zero parameters). Empirical auditing of the trained checkpoint confirms that the network attained syntactic coherence and a held-out validation perplexity of $38.72$ despite being trained on only $\approx 2\%$ of the token volume recommended by compute-optimal Chinchilla heuristics ($N^* \approx 20P$). We analyze the trade-offs of this regime: while extreme quantization and decay act as an inductive regularizer that preserves grammatical structure in low-data settings, it fundamentally restricts the model's capacity for fine-grained factual recall.

---

## 1. Introduction & Theoretical Motivation

Empirical scaling literature demonstrates that unconstrained continuous neural networks allocate substantial parameter capacity to distributed associative memory stores (Geva et al., 2021; Meng et al., 2022). In standard 16-bit floating-point weights, storing idiosyncratic factual co-occurrences requires low gradient energy during early training, which can delay the convergence of compact algorithmic and syntactic circuits (Power et al., 2022).

The PCEI architecture evaluated an explicitly constrained alternative:
$$\mathcal{H}_{\text{capacity}}: \text{Imposing continuous weight decay and discrete ternary quantization restricts high-entropy memorization, compelling the network to prioritize invariant syntactic and structural transitions.}$$

This paper evaluates the empirical training trajectory and representation structure of the 846M-parameter model under this capacity-constrained regime.

---

## 2. Mathematical Formulation of Capacity Constraints

```
                           [ CAPACITY CONSTRAINTS ]
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
   BitNet 1.58b Quantization   Metabolic Decay        Hookean Bond Decay
    W ∈ {-1, 0, +1} · γ       dE/dt = -λ_met · E      dS_ij/dt = -δ_decay
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                                      ▼
                      Restricted Parameter Entropy:
                     H(W) ≤ log2(3) ≈ 1.58 bits/weight
```

### 2.1 Continuous Metabolic Decay
In `src/modules/plasticity.py`, continuous shadow weights are subjected to metabolic decay:
$$\frac{\partial \mathbf{W}_{\text{shadow}}}{\partial t} = -\lambda_{\text{met}} \mathbf{W}_{\text{shadow}} + \eta_{\text{eff}} \nabla_{\mathbf{W}} \mathcal{L}$$
where the effective learning rate $\eta_{\text{eff}}$ is modulated by slow-field confidence:
$$\eta_{\text{eff}}^{(l)} = \eta_{\text{base}} \left(1.0 - \sigma_{\text{conf}}^{(l)} \right)$$

Connections that receive infrequent, low-magnitude gradient updates are driven below the quantization threshold, collapsing into the silent $0$ state.

### 2.2 Discrete Ternary Quantization (BitNet 1.58b)
All linear projections in the 24 text layers employ `TernaryLinear`:
$$W_{ij} \in \left\{ -1, 0, +1 \right\} \cdot \gamma, \quad \gamma = \frac{1}{mn}\sum |W_{ij}|$$
* The maximum information capacity per parameter is strictly upper-bounded by $\log_2(3) \approx 1.585\text{ bits}$.
* Storing arbitrary, rare factual key-value pairs requires configuring thousands of discrete ternary weights. When total parameter capacity is constrained, the model cannot simultaneously memorize extensive trivia and maintain general syntactic grammar, forcing parameter allocation toward recurring structural rules.

### 2.3 Topological Bond Dissolution
Inter-layer bonds in `src/core/broker.py` decay continuously:
$$S_{ij}^{(t+1)} = S_{ij}^{(t)} - \delta_{\text{decay}}$$
Connections without persistent activation co-firing dissolve, preventing stale cross-layer pathways.

---

## 3. Information-Theoretic Capacity Analysis

Under the Information Bottleneck Principle (Tishby & Zaslavsky, 2015), neural learning optimizes the trade-off between compressing input representation $\mathbf{X}$ and preserving predictive mutual information with target $\mathbf{Y}$:
$$\min_{\mathbf{T}} \left[ I(\mathbf{X}; \mathbf{T}) - \beta I(\mathbf{T}; \mathbf{Y}) \right]$$

### 3.1 Parameter Space Contraction
In an unconstrained continuous network with $P$ parameters in FP16, the configuration space possesses volume bounded by $2^{16 P}$.

Under metabolic decay and ternary quantization with steady-state zero-fraction $\zeta = 0.1982$:
$$|\Omega_{\text{reachable}}| \le 3^{P \cdot (1 - \zeta)} = 3^{0.8018 \cdot P} \ll 2^{16 P}$$

Because storing $M$ arbitrary factual associations requires parameter entropy $H \ge \mathcal{O}(M \log V)$, whereas general syntactic transition rules require $H \le \mathcal{O}(D \log 3)$, the capacity constraint restricts brute-force memorization and encourages compact representations.

---

## 4. Empirical Evaluation: Training on a Constrained Token Budget

We audited the production checkpoint (`pcei_750m_reasoning_assistant_latest.pt`) trained on $333.8\text{ million tokens}$:

### Table 1: Training Accounting and Heuristic Baselines
| Metric | Compute-Optimal Allocation (Chinchilla) | PCEI Empirical Budget | Context / Analysis |
| :--- | :--- | :--- | :--- |
| **Target Token Budget ($N^*$)** | $16,925,000,000\text{ tokens}$ | **$333,824,000\text{ tokens}$** | Model trained in low-data regime |
| **Ratio to Compute-Optimal ($N / N^*$)** | $100.0\%$ | **$1.97\%$** | Undertrained relative to Chinchilla |
| **Held-Out Validation Perplexity** | $35.0 – 42.0$ (At full budget) | **$38.72$** (At 1.97% budget) | Reached comparable PPL early |
| **Zero-Weight Sparsity ($\zeta$)** | $0.0\%$ (Dense continuous) | **$19.82\%$** ($65.9\text{M}$ zeros) | Active ternary zero gating |
| **Post-Diffusion PPL Reduction** | — | **$-19.07$ PPL** ($57.79 \to 38.72$) | Offline latent manifold smoothing |

### 4.1 Interpretation of Training Dynamics
1. **Early Syntactic Fluency**: On a budget of $333.8\text{M tokens}$, the model achieved coherent English syntax and a test perplexity of $38.72$. While this does not indicate superior asymptotic performance to full-scale training, it shows that strong inductive biases (ternary quantization, state-space recurrence, and weight decay) can accelerate initial syntactic acquisition.
2. **The $19.82\%$ Zero State**: Across 332.6M ternary parameters, $65,920,368$ parameters remain in state $0$. Metabolic decay successfully pruned low-signal parameters into quiescent gates.
3. **The Memorization Trade-Off**: While the model generalized well syntactically, empirical probing demonstrates that factual recall of obscure entities is substantially lower than in equivalent-scale FP16 models. The capacity constraints enforce structural rules at the direct expense of encyclopedic storage.

---

## 5. Architectural Trade-Offs

1. **Compute-Optimal Scaling Context**: Hoffmann et al. (2022) established that training compute is optimally allocated when tokens scale proportionally to parameters ($N \approx 20P$). Training on $1.97\%$ of this allocation leaves the model compute-suboptimal; substantial language modeling gains remain accessible if trained on larger token corpora.
2. **Edge Hardware Compatibility**: The combination of BitNet 1.58b and metabolic sparsity yields a lightweight weight tensor ($332.6\text{M ternary weights} \approx 65.7\text{ MB}$ raw parameter footprint for linear layers), making the model exceptionally well-suited for edge devices where memory bandwidth is constrained.
3. **Factual Storage Deficit**: Deployments requiring extensive world-knowledge recall must pair this architecture with external retrieval mechanisms (RAG), as the ternary parameter substrate is mathematically ill-suited for memorizing large unstructured databases.

---

## 6. Conclusion

Hypothesis $\mathcal{H}_{\text{capacity}}$ is partially supported: continuous metabolic decay and BitNet 1.58b ternary quantization restrict parameter configuration space, acting as an inductive regularizer that promotes rapid syntactic generalization in low-token regimes. However, this capacity restriction enforces a trade-off: encyclopedic memorization is diminished, underscoring the necessity of external knowledge retrieval for factual question answering.

---

## References

1. Geva, M., et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *Empirical Methods in Natural Language Processing (EMNLP)*.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems*, 35.
3. Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. *Advances in Neural Information Processing Systems*, 35.
4. Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv preprint arXiv:2201.02177*.
5. Tishby, N., & Zaslavsky, N. (2015). Deep Learning and the Information Bottleneck Principle. *IEEE Information Theory Workshop (ITW)*.
6. Wang, H., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *arXiv preprint arXiv:2402.17764*.
