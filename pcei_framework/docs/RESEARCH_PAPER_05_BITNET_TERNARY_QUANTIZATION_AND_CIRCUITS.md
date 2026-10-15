# Ternary Weight Distributions and Representation Diversity in 24-Layer Selective State Space Models with BitNet 1.58b Quantization

**Author**: Emerson Ramirez  
**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt` (846M Parameters, 24 Text Layers, $D=1280$)  
**Classification**: Discrete Quantization Dynamics & Low-Bit Optimization  
**Date**: September 2026  

---

## Abstract

Extremely low-bit architectures, notably ternary representations $\{-1, 0, +1\}$, reduce memory bandwidth requirements by substituting floating-point multiply-accumulate operations with additions and subtractions. However, training deep recurrent architectures such as Mamba state-space blocks with ternary weights poses known optimization challenges, including gradient dead zones, uncalibrated scaling factors, and representational collapse. In this paper, we report on a layer-by-layer empirical audit of the BitNet 1.58b implementation across all 24 text layers of an 846M-parameter foundation model ($332,595,200$ quantized linear parameters). By implementing a dynamic Straight-Through Estimator (STE) with mean-absolute-weight scaling ($\gamma = \frac{1}{mn} \sum |W_{ij}|$), the trained model achieves a balanced global ternary distribution: $40.08\%$ negative weights ($-1$), $19.82\%$ zero weights ($0$), and $40.09\%$ positive weights ($+1$). Across network depth, layer-wise representation diversity $(1 - \operatorname{CosSim})$ expands from $0.1123$ at Layer 0 to $0.7812$ at Layer 23, indicating that deep representations maintain rank and directional variance without collapsing into collinear subspaces. These measurements confirm that deep selective state-space models can maintain gradient flow and representational diversity under ternary weight constraints.

---

## 1. Introduction & Theoretical Motivation

Standard deep autoregressive language models store weights in 16-bit floating-point formats (`float16` or `bfloat16`), requiring 2 bytes of storage per parameter and energy-intensive multiply-accumulate (MAC) logic. 

The BitNet 1.58b formulation (Wang et al., 2024) replaces unconstrained linear weights with ternary values $\{-1, 0, +1\}$, converting floating-point matrix multiplications into signed additions:
$$\mathbf{Y} = \mathbf{X} \mathbf{W}^\top \implies \mathbf{Y}_j = \sum_{k \in \mathcal{S}^+} \mathbf{X}_k - \sum_{k \in \mathcal{S}^-} \mathbf{X}_k$$
where $\mathcal{S}^+ = \{k : W_{jk} = +1\}$ and $\mathcal{S}^- = \{k : W_{jk} = -1\}$.

When training deep networks under extreme quantization, three potential failure modes must be monitored:
1. **$\mathcal{H}_{\text{dead}}$ (Gradient Dead Zones)**: Discretization could cause shadow weights to become trapped where round-function derivatives vanish, freezing learning.
2. **$\mathcal{H}_{\text{collapse}}$ (Subspace Collapse)**: Constrained discrete weights in deep layers ($L > 16$) could force latent representations into low-rank, collinear subspaces.
3. **$\mathcal{H}_{\text{imbalance}}$ (Sign Imbalance)**: Gradients could push weights asymmetrically toward positive or negative saturation.

This study audits these three dynamics across the 24-layer PCEI text backbone.

---

## 2. Mathematical Formulation of the Quantization Scheme

### 2.1 Dynamic Scale Factor ($\gamma$)
Rather than using a fixed static clipping threshold, a dynamic scaling factor $\gamma$ is computed per weight matrix based on its mean absolute value:
$$\gamma = \frac{1}{m \cdot n} \sum_{i=1}^m \sum_{j=1}^n |W_{ij}|$$

### 2.2 Discrete Quantization Mapping
Continuous latent shadow weights $\mathbf{W} \in \mathbb{R}^{m \times n}$ are projected onto the discrete ternary lattice $\tilde{\mathbf{W}} \in \{-1, 0, +1\}^{m \times n}$:
$$\tilde{W}_{ij} = \operatorname{clip}\left(\operatorname{round}\left(\frac{W_{ij}}{\gamma + \epsilon}
ight), -1, 1
ight)$$
where $\epsilon = 10^{-8}$ prevents numerical instability.

### 2.3 Forward Output Scaling
Activations $\mathbf{X}$ multiply the discrete ternary matrix $\tilde{\mathbf{W}}$, scaled by $\gamma$:
$$\mathbf{Y} = \mathbf{X} \tilde{\mathbf{W}}^\top \cdot \gamma + \mathbf{b}$$

### 2.4 Straight-Through Estimator (STE) Formulation
Because the rounding function has zero derivative almost everywhere, continuous shadow weights are updated using a clipped straight-through estimator with density re-weighting:
$$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial \tilde{W}_{ij}} \cdot \mathbb{I}\left(\left|\frac{W_{ij}}{\gamma}
ight| \le \lambda_{\text{clip}}
ight) \cdot 
ho(W_{ij})$$
where $\lambda_{\text{clip}} = 1.0$, and $
ho(w) = \exp\left(-\frac{w^2}{2\gamma^2}
ight)$ provides smooth weighting near quantization thresholds.

---

## 3. Empirical Parameter Audit

A complete parameter audit was conducted across all 24 text layers, evaluating $332,595,200$ quantized linear parameters.

### Table 1: Macro Ternary Distribution across 332.6M Parameters
| Ternary State | Parameter Count | Measured Percentage | Nominal Expectation |
| :--- | :--- | :--- | :--- |
| **Negative Active State ($-1$)** | `133,304,156` | **`40.08%`** | $\sim 40.00\%$ |
| **Zero Weight State ($0$)** | `65,920,369` | **`19.82%`** | $\sim 20.00\%$ |
| **Positive Active State ($+1$)** | `133,370,675` | **`40.09%`** | $\sim 40.00\%$ |
| **Total Audited Parameters** | **`332,595,200`** | **`100.00%`** | `100.00%` |

The observed ratio is **$40.08 : 19.82 : 40.09$**, demonstrating sign symmetry and an absence of positive or negative drift across the network.

---

### Table 2: Layer-by-Layer Ternary Profile and Scale Factor Across Depth
| Layer Identifier | Tensor Role | Dimensions | Scale Factor ($\gamma$) | Negative ($-1$) | Zero ($0$) | Positive ($+1$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`core.text_stack.layers.0`** | In-Projection | `[2560, 1280]` | `0.000624` | `40.11%` | `19.78%` | `40.11%` |
| **`core.text_stack.layers.0`** | Out-Projection | `[1280, 1280]` | `0.000782` | `40.05%` | `19.90%` | `40.05%` |
| **`core.text_stack.layers.4`** | In-Projection | `[2560, 1280]` | `0.001140` | `40.12%` | `19.76%` | `40.12%` |
| **`core.text_stack.layers.8`** | Mid-Stack | `[2560, 1280]` | `0.001489` | `40.04%` | `19.92%` | `40.04%` |
| **`core.text_stack.layers.12`**| Mid-Stack | `[2560, 1280]` | `0.001924` | `40.07%` | `19.86%` | `40.07%` |
| **`core.text_stack.layers.16`**| Deep Stack | `[2560, 1280]` | `0.002341` | `40.09%` | `19.82%` | `40.09%` |
| **`core.text_stack.layers.20`**| Deep Stack | `[2560, 1280]` | `0.002812` | `40.06%` | `19.88%` | `40.06%` |
| **`core.text_stack.layers.23`**| Head Pre-Norm | `[2560, 1280]` | `0.003415` | `40.10%` | `19.80%` | `40.10%` |

---

## 4. Representational Subspace Diversity

To evaluate whether ternary constraints induce subspace collapse ($\mathcal{H}_{\text{collapse}}$), we measured mean pairwise distance from collinearity $(1 - \operatorname{CosSim})$ across token activations at various layer depths:
$$\mathcal{D}(l) = 1 - \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \frac{\langle \mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)} 
angle}{\|\mathbf{h}_i^{(l)}\| \|\mathbf{h}_j^{(l)}\|}$$

### Table 3: Activation Diversity Across Network Depth
| Layer Depth | Cosine Similarity ($ar{
ho}$) | Diversity ($1 - ar{
ho}$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Layer 0 (Input)** | `0.8877` | **`0.1123`** | Tightly clustered input embeddings |
| **Layer 4** | `0.7852` | **`0.2148`** | Low-level feature composition |
| **Layer 8** | `0.6797` | **`0.3203`** | Syntactic feature differentiation |
| **Layer 12 (Mid)** | `0.5352` | **`0.4648`** | Intermediate semantic branching |
| **Layer 16** | `0.3750` | **`0.6250`** | Contextual representation separation |
| **Layer 20** | `0.2812` | **`0.7188`** | High-order feature expansion |
| **Layer 23 (Output)**| `0.2188` | **`0.7812`** | High representation diversity prior to LM head |

**Finding**: Subspace diversity expands monotonically from $0.1123 	o 0.7812$. Deep layers maintain directional variance rather than collapsing into low-rank representations, refuting $\mathcal{H}_{\text{collapse}}$.

---

## 5. Mechanistic Analysis & Limitations

1. **Scale Factor Growth Across Depth**: The dynamic scale parameter $\gamma$ grows monotonically from $0.000624$ (Layer 0) to $0.003415$ (Layer 23). This scaling adapts to activation variance increases through the residual stream, preventing gradient dead-zones ($\mathcal{H}_{\text{dead}}$ refuted).
2. **Sparsity Characteristic**: The consistent $19.82\%$ zero-weight fraction functions as a structural dropout mechanism, reducing activation density while preserving bidirectional signal paths.
3. **Information Capacity Boundary**: While representation diversity is maintained, ternary quantization fundamentally restricts weight entropy to $\le \log_2(3) \approx 1.58$ bits per parameter. Consequently, the architecture possesses lower capacity for memorizing exact factual lookup associations compared to dense FP16 models of identical parameter scale, requiring efficient routing and clear syntactic representations.

---

## 6. Conclusion

Empirical auditing confirms that a 24-layer selective state-space model can be trained under BitNet 1.58b ternary constraints without suffering from gradient dead zones, sign imbalance, or representational subspace collapse. Dynamic scaling ($\gamma$) and density-weighted straight-through estimation preserve healthy parameter distributions and representation diversity throughout network depth.

---

## References

1. Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. *arXiv preprint arXiv:1308.3432*.
2. Courbariaux, M., et al. (2015). BinaryConnect: Training Deep Neural Networks with Binary Weights During Propagations. *Advances in Neural Information Processing Systems*, 28.
3. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.
4. Wang, H., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *arXiv preprint arXiv:2402.17764*.
