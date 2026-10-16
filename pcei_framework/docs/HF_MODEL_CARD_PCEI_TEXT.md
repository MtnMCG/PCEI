---
language:
- en
license: mit
tags:
- pcei
- bitnet
- ternary
- 1.58bit
- mamba
- ssm
- state-space-model
- fastkan
- linear-attention
- energy-efficient
- green-ai
pipeline_tag: text-generation
---

# PCEI-Text: Physically Constrained Emergent Intelligence (100M BitNet Ternary Mamba)

[![GitHub Repository](https://img.shields.io/badge/GitHub-MtnMCG%2FPCEI-blue.svg)](https://github.com/MtnMCG/PCEI)
[![Sister Repo: PCEI-Robotics](https://img.shields.io/badge/🤗%20HuggingFace-PCEI--Robotics-blueviolet.svg)](https://huggingface.co/MtnMCG/PCEI-Robotics)
[![Architecture: Selective SSM + BitNet 1.58b](https://img.shields.io/badge/Architecture-Mamba%20SSM%20%2B%20BitNet%201.58b-red.svg)]()
[![Complexity: O(L) Linear](https://img.shields.io/badge/Inference%20Complexity-O(1)%20Memory%20%7C%20O(L)%20Compute-green.svg)]()
[![Quantization: 1.58-bit {-1, 0, +1}](https://img.shields.io/badge/Quantization-Strict%20BitNet%20b1.58-brightgreen.svg)]()

> **Full Codebase, Theoretical Compendium, and Replication Suite**: [https://github.com/MtnMCG/PCEI](https://github.com/MtnMCG/PCEI)

---

## 🏛️ Executive Summary

**PCEI-Text** is an open-weights linear-time sequence model that discards quadratic Transformer attention ($\mathcal{O}(L^2)$) and high-energy 16-bit floating-point multiplications in favor of:
1. **Hardware-efficient Selective State Spaces (Mamba SSM)** operating with linear $\mathcal{O}(L)$ compute and constant $\mathcal{O}(1)$ memory inference.
2. **Canonical Microsoft BitNet b1.58 Ternary Quantization**, constraining synaptic weights to strictly discrete $\{-1.0, 0.0, +1.0\}$ states and replacing energy-hungry floating-point multiplications ($FMUL$) with integer additions ($IADD$).
3. **Kolmogorov-Arnold Network (FastKAN) Spline Manifolds**, utilizing learnable cubic B-splines along non-linear parameter trajectories.
4. **Dual-Timescale Dynamic State Modulation (`AmbientField`)**, preventing context degradation and drifting attractors over extended multi-turn horizons.

This repository hosts foundation checkpoint `pcei_100m_bitnet_ternary.pt` (1.09 GB), trained continuously on diverse natural language, dialogue, reasoning, and synthetic curricula.

---

## 🔬 Empirical Audits & Verification

### 1. Tiny LM Compute-Matched Bake-Off (Muse Autonomous Evaluation)
In an independent multi-seed compute-matched baseline bake-off conducted via the Muse evaluation agent against a compute-matched Transformer:

| Model | Architecture | Parameters | Val Loss (mean ± std) | Val Perplexity | Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Transformer** | Vanilla Decoder (5L, 4H, FFN $2\times d$ GELU) | 398,688 | 4.6131 ± 0.0003 | 100.8 | Baseline |
| **Dense Mamba** | Repo `MambaBlock` (6L, FP32 dense weights) | 407,232 | 4.3881 ± 0.0009 | 80.5 | 0 / 2 vs PCEI |
| **PCEI-tiny** | **Strict 1.58-bit Ternary (`{-1, 0, +1}`) + FastKAN** | 411,912 | **4.3783 ± 0.0073** | **79.1** | **2 / 2 (Won all seeds)** |

* **Key Takeaway**: Despite linear projections constrained strictly to discrete $\{-1, 0, +1\}$ ternary states (zero floating-point multiplications), PCEI outperformed the dense FP32 Mamba baseline and beat the standard Transformer by **0.23 nats** (val perplexity 79.1 vs 100.8).

### 2. Strict Microsoft BitNet b1.58 Migration
An external audit in October 2026 disclosed that early internal prototypes utilized an experimental 5-state quantizer. The architecture was fully refactored to canonical Microsoft BitNet b1.58:
- **Parameter Entropy Cut**: Reduced from 2.32b to strictly 1.58b ($\le \log_2(3)$ bits/param), cutting parameter entropy by **32%**.
- **Perplexity & Hallucinations**: Validation loss improved from 11.1696 to 11.1429 on 100M, eliminating activation outlier code token hallucinations.
- Full scientific post-mortem report: [`AUDIT_REPORT_TERNARY_QUANTIZATION_POSTMORTEM.md`](https://github.com/MtnMCG/PCEI/blob/main/pcei_framework/docs/AUDIT_REPORT_TERNARY_QUANTIZATION_POSTMORTEM.md).

---

## ⚙️ Model Specifications

| Architectural Dimension | Specification |
| :--- | :--- |
| **Base Architecture** | 12-Layer Selective State Space (Mamba SSM) |
| **Model Dimension ($d_{\text{model}}$)** | 768 |
| **Inner Expansion Dimension ($d_{\text{inner}}$)** | 512 |
| **State Dimension ($d_{\text{state}}$)** | 16 |
| **Convolution Kernel Size** | 4 |
| **Non-Linear Branch** | FastKAN (Learnable cubic B-splines, grid size = 5) |
| **Synaptic Quantization** | Strict BitNet b1.58 ($\{-1.0, 0.0, +1.0\}$) via STE |
| **Vocabulary Size** | 50,257 (OpenAI GPT-2 BPE tokenizer) |
| **Context Length ($L$)** | 128 / 512 / 2048 streaming tokens |
| **Inference Footprint** | $\approx 250\text{ MB}$ packed ternary integer weights |
| **Hardware Target** | Apple Silicon MPS, NVIDIA CUDA, or CPU edge devices |

---

## 🚀 Local Inference Quickstart

To run local text generation using the PCEI framework:

```bash
# 1. Clone the master GitHub repository
git clone https://github.com/MtnMCG/PCEI.git
cd PCEI/pcei_framework

# 2. Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub

# 3. Download the model checkpoint directly from Hugging Face
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='MtnMCG/PCEI-Text', filename='pcei_100m_bitnet_ternary.pt', local_dir='checkpoints')
"

# 4. Interactive generation with continuous state modulation
python3 tools/resonance_chat.py \
  --checkpoint checkpoints/pcei_100m_bitnet_ternary.pt \
  --device mps  # Use --device cuda on NVIDIA GPUs
```

### Python API Usage

```python
import torch
from transformers import GPT2Tokenizer
from src.mamba_text import MambaConfig, MambaTextStack

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Initialize architecture
config = MambaConfig(
    d_model=768,
    d_inner=512,
    n_layer=12,
    vocab_size=50257,
    use_bitnet=True,
    use_fastkan=True
)
model = MambaTextStack(config).to(device)

# Load checkpoint
checkpoint = torch.load("checkpoints/pcei_100m_bitnet_ternary.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()

# Encode prompt
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("The fundamental principle of energy efficiency in neural computation", return_tensors="pt").to(device)

# Autoregressive inference
with torch.no_grad():
    for _ in range(64):
        logits, _ = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)

print(tokenizer.decode(tokens[0]))
```

---

## 🤖 Sister Repositories & Embodied Intelligence

PCEI proves that physical constraints generalize across domains:
- 🏃 **[MtnMCG/PCEI-Robotics](https://huggingface.co/MtnMCG/PCEI-Robotics)**: Autonomous American Ninja Warrior 18-meter humanoid controller traversing 45° angled Quad Steps without attention or deep RL.
- 🏛️ **[PCEI Master GitHub Repository](https://github.com/MtnMCG/PCEI)**: Contains the 11 theoretical research papers, continuous training daemons, curriculum streaming pipelines, and full verification suite.

---

## 📜 Citation & License

This model and the PCEI codebase are released under the **MIT License**.

```bibtex
@article{ramirez2026pcei,
  title   = {Physically Constrained Emergent Intelligence: Subverting Transformer Hyperscaling via Linear State Spaces, Ternary Synaptic Gating, and Latent Generative Replay},
  author  = {Ramirez, Emerson and Antigravity Assistant},
  journal = {PCEI Theoretical Research Compendium},
  year    = {2026},
  url     = {https://github.com/MtnMCG/PCEI}
}
```
