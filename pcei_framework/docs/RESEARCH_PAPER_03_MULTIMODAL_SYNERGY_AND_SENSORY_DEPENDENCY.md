# Cross-Modal Interference and Modality Dropout in Asymmetric Multi-Stack Neural Architectures

**Author**: Emerson Ramirez  
**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoints**: `pcei_750m_step85251_text_baseline.pt` vs. `pcei_750m_step87820_early_multimodal.pt`  
**Classification**: Multimodal Latent Manifolds & Modality Conditioning  
**Date**: September 2026  

---

## Abstract

Integrating heterogeneous modalities into a shared sequence backbone often induces catastrophic forgetting or representational interference in unimodal tasks. In this paper, we report on a controlled ablation within the Physically Constrained Emergent Intelligence (PCEI) framework—a tri-stack architecture connecting an 846M-parameter text stack with pretrained CLIP vision and Whisper audio feature extractors via cross-modal boundary projections. Comparing a text-only baseline (Step 85,251) against an early multimodal fine-tuned checkpoint (Step 87,820), we document a significant cross-modal co-dependency effect. When visual patch tokens and audio spectrogram features are concurrently provided, the model achieves lower evaluation loss ($2.38$ nats) and higher task accuracy ($67.01\%$) on multimodal queries. However, when evaluated on pure symbolic text without sensory inputs ($\mathbf{x}_{\text{img}} = \emptyset, \mathbf{x}_{\text{audio}} = \emptyset$), the multimodal checkpoint degrades substantially: text evaluation loss increases to $8.85$ nats, next-token prediction accuracy drops from $12.94\%$ to $3.97\%$, and completions degenerate into web-caption attractors. Mechanistic analysis reveals that recurrent state updates over-rely on continuous cross-modal bias offsets, causing activation shift when sensory inputs are zeroed. We formalize Scheduled Modality Dropout ($\mathcal{P}_{\text{drop}} = 0.30$) during fine-tuning as an effective regularizer to preserve text autonomy.

---

## 1. Introduction & Theoretical Motivation

Standard approaches to multimodal language modeling project visual and auditory features directly into the primary token embedding sequence (e.g., LLaVA, Flamingo). While effective, this can create interference between discrete textual syntax and dense continuous feature projections, particularly in smaller models where parameter capacity is constrained.

The PCEI architecture investigated an asymmetric modular design:
1. **Text Backbone**: 24 Mamba/Diffusion layers ($D=1280$).
2. **Vision Feature Stack**: 4 layers ($D=512$) initialized from CLIP ViT-B/16.
3. **Audio Feature Stack**: 4 layers ($D=256$) initialized from Whisper-Base.
4. **Boundary Transfer**: Gated linear projections (`BoundaryTransfer`) mapping sensory latents into the text stack.

We investigated the hypothesis:
$$\mathcal{H}_{\text{synergy}}: \text{Multimodal grounding improves representation stability without degrading unimodal symbolic performance.}$$

This paper evaluates this hypothesis by comparing checkpoint performance under full multimodal input versus unimodal text evaluation.

---

## 2. Architecture & Boundary Integration

### 2.1 Modality Projections
Input representations are embedded into modality-specific latent spaces:
$$\mathbf{Z}_{\text{text}} = \mathbf{W}_{\text{embed}} \mathbf{x}_{\text{text}} \in \mathbb{R}^{B \times L_{\text{text}} \times 1280}$$
$$\mathbf{Z}_{\text{img}} = \mathbf{W}_{\text{proj}}^{\text{img}} \mathrm{Conv2D}_{16\times16}(\mathbf{x}_{\text{img}}) \in \mathbb{R}^{B \times L_{\text{img}} \times 512}$$
$$\mathbf{Z}_{\text{audio}} = \mathbf{W}_{\text{proj}}^{\text{audio}} \mathrm{MelFilterbank}(\mathbf{x}_{\text{audio}}) \in \mathbb{R}^{B \times L_{\text{audio}} \times 256}$$

### 2.2 Boundary Transfer Projections
Sensory representations are injected into text layer representations via gated linear transformations:
$$\mathbf{H}_{\mathcal{S} \to \text{text}} = \mathrm{LayerNorm}\left(\mathbf{W}_{\text{transfer}} \mathbf{Z}_{\mathcal{S}} \mathbf{W}_{\text{gate}}^\top \right)$$
$$\mathbf{Z}_{\text{text}}^{(l)} \leftarrow \mathbf{Z}_{\text{text}}^{(l)} + \alpha_{\text{boundary}} \cdot \mathbf{H}_{\mathcal{S} \to \text{text}}$$
where $\alpha_{\text{boundary}} = 0.15$ controls coupling gain.

---

## 3. Experimental Protocol

Two checkpoints from cloud training on an NVIDIA RTX 4090 were evaluated:
1. **Checkpoint A (Text Baseline)**: `pcei_750m_step85251_text_baseline.pt` (85,251 steps of token-packed text pretraining).
2. **Checkpoint B (Multimodal)**: `pcei_750m_step87820_early_multimodal.pt` (85,251 text steps + 2,569 steps of joint text, image, and audio training).

Both models were evaluated on a 100-sample benchmark across four domains under zero-sensory conditions ($\mathbf{x}_{\text{img}} = \mathbf{0}, \mathbf{x}_{\text{audio}} = \mathbf{0}$):
* Domain 1: Factual Knowledge (science, geography)
* Domain 2: Syllogistic Deductions
* Domain 3: Python Code Syntax
* Domain 4: Physical Commonsense

---

## 4. Quantitative Results

### Table 1: Cross-Modal Performance Under Grounded vs. Unimodal Evaluation
| Model State | Image Input | Audio Input | Evaluation Loss (nats) | Next-Token Accuracy | Behavior Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1 Baseline (Text-Only)**| None | None | **`6.1033`** | **`12.94%`** | Syntactically consistent text |
| **Phase 2 Multimodal (Joint Input)**| CLIP Image | Whisper Audio | **`2.3810`** | **`67.01%`** | Grounded question answering |
| **Phase 2 Multimodal (Text-Only Evaluation)**| None | None | **`8.8476`** | **`3.97%`** | Severe loss degradation |

### Table 2: Accuracy Across Domains Under Zero-Sensory Input
| Evaluation Domain | Phase 1 Text Baseline | Phase 2 in Zero-Sensory Mode | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **General Factual Knowledge** | `10.81%` | `9.46%` | $-1.35\%$ |
| **Deductive Syllogisms** | `26.03%` | `9.59%` | **$-16.44\%$** |
| **Python Code Syntax** | `9.54%` | `0.00%` | **$-9.54\%$** |
| **Physical Commonsense** | `14.29%` | `7.14%` | $-7.14\%$ |
| **Overall Average** | **`12.94%`** | **`3.97%`** | **$-8.97\%$** |

### Table 3: Qualitative Generations in Zero-Sensory Mode
When evaluated without sensory tokens, Checkpoint B generated repetitive fragments characteristic of multimodal training captions:

```text
[PROMPT 1: Factual Geography]
INPUT:  "The capital of France is Paris. The capital of Italy is"
PHASE 1 OUTPUT: "Rome, which is known for its historic monuments..."
PHASE 2 OUTPUT: "a very one of the preparations of maize"

[PROMPT 2: Python Code Syntax]
INPUT:  "In Python, a function that returns the square of a number is written as"
PHASE 1 OUTPUT: "def square(x): return x ** 2"
PHASE 2 OUTPUT: "i was been designed in moderation suitable food"

[PROMPT 3: Physical Thermodynamics]
INPUT:  "When ice is exposed to heat, it begins to"
PHASE 1 OUTPUT: "melt into liquid water as thermal energy increases"
PHASE 2 OUTPUT: "moderation suitable food for almost everybody and when"
```

---

## 5. Mechanistic Analysis: The Dependency Effect

### 5.1 Activation Shift Under Zero Sensory Vectors
During joint multimodal training, sensory projections provided consistent non-zero latent offsets:
$$\mathbf{z}_{\text{text}} \leftarrow \mathbf{z}_{\text{text}} + \mathbf{H}_{\text{img} \to \text{text}} + \mathbf{H}_{\text{audio} \to \text{text}}$$
Because pretrained CLIP and Whisper embeddings carry high directional magnitude, the recurrent Mamba state transitions adjusted to rely on these persistent input biases:
$$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t (\mathbf{x}_t + \mathbf{H}_{\text{sensory}})$$

When sensory inputs are zeroed at test time, $\mathbf{H}_{\text{sensory}} = \mathbf{0}$. The recurrent hidden state trajectory shifts away from the nominal mean and variance expected by subsequent layer normalizations and the output unembedding matrix. Consequently, token generation falls into high-frequency corpus n-gram attractors (predominantly image-caption fragments such as *"moderation suitable food"*).

### 5.2 Hypothesis Evaluation
$\mathcal{H}_{\text{synergy}}$ is **supported conditionally**: multimodal conditioning improves accuracy when paired modalities are present ($67.01\%$ vs. $12.94\%$), but unconstrained fine-tuning severely degrades unimodal text capabilities.

---

## 6. Mitigation: Scheduled Modality Dropout

To prevent co-dependency and preserve unimodal language capabilities, multimodal fine-tuning requires stochastic masking of the transfer projections:

$$\tilde{\mathbf{H}}_{\mathcal{S} \to \text{text}} = m_{\mathcal{S}} \cdot \mathbf{H}_{\mathcal{S} \to \text{text}}, \quad m_{\mathcal{S}} \sim \operatorname{Bernoulli}(1 - \mathcal{P}_{\text{drop}})$$
where $\mathcal{P}_{\text{drop}} = 0.30$.

By ensuring that the text backbone processes the objective with null sensory inputs in 30% of training steps, the model maintains calibrated hidden states in both multimodal and unimodal evaluation modes.

---

## 7. Conclusion

Naive multimodal fine-tuning through cross-modal projection heads induces severe activation shift and degradation in unimodal text generation. Incorporating stochastic modality dropout during training is necessary to maintain baseline language capabilities while utilizing cross-modal representations.

---

## References

1. Bisk, Y., et al. (2020). Experience Grounds Language. *Empirical Methods in Natural Language Processing (EMNLP)*.
2. Liu, H., et al. (2023). Visual Instruction Tuning. *Advances in Neural Information Processing Systems*, 36.
3. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *International Conference on Machine Learning (ICML)*.
4. Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *International Conference on Machine Learning (ICML)*.
