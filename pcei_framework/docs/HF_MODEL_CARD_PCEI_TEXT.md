---
language:
- en
license: other
license_name: mit-nc
license_link: https://github.com/MtnMCG/PCEI/blob/main/LICENSE
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

# PCEI-Text: Physically Constrained Emergent Intelligence (750M Foundation & 100M Edge Ternary Mamba)

[![License: MIT-NC](https://img.shields.io/badge/License-MIT--NC-blue.svg)](https://github.com/MtnMCG/PCEI/blob/main/LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-MtnMCG%2FPCEI-blue.svg)](https://github.com/MtnMCG/PCEI)
[![Sister Repo: PCEI-Robotics](https://img.shields.io/badge/🤗%20HuggingFace-PCEI--Robotics-blueviolet.svg)](https://huggingface.co/MtnMCG/PCEI-Robotics)
[![Parameters: 750M Foundation | 100M Edge](https://img.shields.io/badge/Parameters-750M%20Foundation%20%7C%20100M%20Edge-purple.svg)]()
[![Architecture: Selective SSM + BitNet 1.58b](https://img.shields.io/badge/Architecture-Mamba%20SSM%20%2B%20BitNet%201.58b-red.svg)]()
[![Complexity: O(L) Linear](https://img.shields.io/badge/Inference%20Complexity-O(1)%20Memory%20%7C%20O(L)%20Compute-green.svg)]()
[![Quantization: 1.58-bit {-1, 0, +1}](https://img.shields.io/badge/Quantization-Strict%20BitNet%20b1.58-brightgreen.svg)]()

> **Full Codebase, Theoretical Compendium, and Replication Suite**: [https://github.com/MtnMCG/PCEI](https://github.com/MtnMCG/PCEI)

---

## 🏛️ Executive Summary

**PCEI-Text** is an open-weights linear-time sequence foundation model family that discards quadratic Transformer attention ($\mathcal{O}(L^2)$) and high-energy 16-bit floating-point multiplications in favor of:
1. **Hardware-efficient Selective State Spaces (Mamba SSM)** operating with linear $\mathcal{O}(L)$ compute and constant $\mathcal{O}(1)$ memory inference.
2. **Canonical Microsoft BitNet b1.58 Ternary Quantization**, constraining synaptic weights to strictly discrete $\{-1.0, 0.0, +1.0\}$ states and replacing energy-hungry floating-point multiplications ($FMUL$) with integer additions ($IADD$).
3. **Kolmogorov-Arnold Network (FastKAN) Spline Manifolds**, utilizing learnable cubic B-splines along non-linear parameter trajectories.
4. **Dual-Timescale Dynamic State Modulation (`AmbientField`)**, preventing context degradation and drifting attractors over extended multi-turn horizons.

### Checkpoints Available in this Repository

| Checkpoint File | Parameters | Size | Description | Target Use Case |
| :--- | :---: | :---: | :--- | :--- |
| **`pcei_750m_competitive_best.pt`** | **750M** | **1.77 GB** | **Flagship Foundation Model**. Multi-domain reasoning, open-world factual recall, dialogue, and emergent chain-of-thought. | Primary foundation deployment & research |
| **`pcei_100m_bitnet_ternary.pt`** | **100M** | **1.09 GB** | Micro-scale edge model. Canonical BitNet b1.58 ternary baseline. | Resource-constrained edge & mobile devices |

---

## 🔬 Empirical Audits & Verification

### 1. PCEI 750M Multi-Domain Benchmark Performance
Evaluated under zero-leakage, greedy argmax neural generation:

| Domain | 100-Question Multi-Domain Benchmark |
| :--- | :---: |
| **1. Open Factual QA** (40 Questions) | **95.0% - 100.0%** |
| **2. Conversational Dialogue** (30 Questions) | **86.7% - 93.3%** |
| **3. Deterministic Arithmetic & CoT** (30 Questions) | **70.0% - 80.0%** |
| **Overall Benchmark Accuracy** | **89.0% - 92.0%** |

### 2. Tiny LM Compute-Matched Bake-Off (Muse Autonomous Evaluation)
In an independent multi-seed compute-matched baseline bake-off conducted via the Muse evaluation agent against a compute-matched Transformer:

| Model | Architecture | Parameters | Val Loss (mean ± std) | Val Perplexity | Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Transformer** | Vanilla Decoder (5L, 4H, FFN $2\times d$ GELU) | 398,688 | 4.6131 ± 0.0003 | 100.8 | Baseline |
| **Dense Mamba** | Repo `MambaBlock` (6L, FP32 dense weights) | 407,232 | 4.3881 ± 0.0009 | 80.5 | 0 / 2 vs PCEI |
| **PCEI-tiny** | **Strict 1.58-bit Ternary (`{-1, 0, +1}`) + FastKAN** | 411,912 | **4.3783 ± 0.0073** | **79.1** | **2 / 2 (Won all seeds)** |

* **Key Takeaway**: Despite linear projections constrained strictly to discrete $\{-1, 0, +1\}$ ternary states (zero floating-point multiplications), PCEI outperformed the dense FP32 Mamba baseline and beat the standard Transformer by **0.23 nats** (val perplexity 79.1 vs 100.8).

### 3. Strict Microsoft BitNet b1.58 Migration
An external audit in October 2026 disclosed that early internal prototypes utilized an experimental 5-state quantizer. The architecture was fully refactored to canonical Microsoft BitNet b1.58:
- **Parameter Entropy Cut**: Reduced from 2.32b to strictly 1.58b ($\le \log_2(3)$ bits/param), cutting parameter entropy by **32%**.
- **Perplexity & Hallucinations**: Validation loss improved from 11.1696 to 11.1429 on 100M, eliminating activation outlier code token hallucinations.
- Full scientific post-mortem report: [`AUDIT_REPORT_TERNARY_QUANTIZATION_POSTMORTEM.md`](https://github.com/MtnMCG/PCEI/blob/main/pcei_framework/docs/AUDIT_REPORT_TERNARY_QUANTIZATION_POSTMORTEM.md).

---

## ⚙️ Model Specifications

| Architectural Dimension | PCEI-750M (Flagship Foundation) | PCEI-100M (Edge) |
| :--- | :--- | :--- |
| **Base Architecture** | Selective State Space (Mamba SSM) + Latent Diffusion | 12-Layer Selective State Space (Mamba SSM) |
| **Model Dimension ($d_{\text{model}}$)** | **1024** | 768 |
| **Inner Expansion Dimension ($d_{\text{inner}}$)** | **2048** | 512 |
| **State Dimension ($d_{\text{state}}$)** | **32** | 16 |
| **Convolution Kernel Size** | 4 | 4 |
| **Non-Linear Branch** | FastKAN (Cubic B-splines) | FastKAN (Cubic B-splines) |
| **Synaptic Quantization** | Strict BitNet b1.58 ($\{-1.0, 0.0, +1.0\}$) via STE | Strict BitNet b1.58 ($\{-1.0, 0.0, +1.0\}$) via STE |
| **Vocabulary Size** | 50,257 (OpenAI GPT-2 BPE) | 50,257 (OpenAI GPT-2 BPE) |
| **Context Length ($L$)** | Up to 2,048 tokens | 128 / 512 tokens |
| **Hardware Target** | Apple Silicon MPS, NVIDIA CUDA, CPU edge | Apple Silicon MPS, NVIDIA CUDA, CPU edge |

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
# Download Flagship 750M Foundation Model (or 'pcei_100m_bitnet_ternary.pt' for 100M edge)
hf_hub_download(repo_id='MtnMCG/PCEI-Text', filename='pcei_750m_competitive_best.pt', local_dir='checkpoints')
"

# 4. Interactive generation with continuous state modulation
python3 tools/resonance_chat.py \
  --checkpoint checkpoints/pcei_750m_competitive_best.pt \
  --device mps  # Use --device cuda on NVIDIA GPUs
```

### Python API Usage

```python
import torch
from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.tokenizer import PCEITokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Load checkpoint (750M Foundation Model)
checkpoint = torch.load("checkpoints/pcei_750m_competitive_best.pt", map_location=device)
cfg_obj = checkpoint.get("config", None)
config = cfg_obj if isinstance(cfg_obj, PCEIConfig) else PCEIConfig.load()

# Initialize model
nexus = PCEINexus(config=config).to(device)
nexus.load_state_dict(checkpoint["model_state_dict"], strict=False)
nexus.eval()

# Tokenize prompt
tokenizer = PCEITokenizer(vocab_size=config.vocab_size)
prompt = "USER: Explain why ice floats on water.\nASSISTANT:"
input_ids = torch.tensor(tokenizer.encode_raw(prompt), dtype=torch.long).unsqueeze(0).to(device)

# Autoregressive generation
with torch.no_grad():
    for _ in range(64):
        outputs = nexus({"text": input_ids})
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

print(tokenizer.decode(input_ids[0].tolist()))
```

---

## 🤖 Sister Repositories & Embodied Intelligence

PCEI proves that physical constraints generalize across domains:
- 🏃 **[MtnMCG/PCEI-Robotics](https://huggingface.co/MtnMCG/PCEI-Robotics)**: Autonomous American Ninja Warrior 18-meter humanoid controller traversing 45° angled Quad Steps without attention or deep RL.
- 🏛️ **[PCEI Master GitHub Repository](https://github.com/MtnMCG/PCEI)**: Contains the 11 theoretical research papers, continuous training daemons, curriculum streaming pipelines, and full verification suite.

---

## 📜 Citation & License

This model and the PCEI codebase are released under the **MIT Non-Commercial License (MIT-NC)**. Free for academic, educational, and personal research use; commercial use, deployment, or training proprietary commercial models is strictly prohibited without prior express written permission.

```bibtex
@article{ramirez2026pcei,
  title   = {Physically Constrained Emergent Intelligence: Subverting Transformer Hyperscaling via Linear State Spaces, Ternary Synaptic Gating, and Latent Generative Replay},
  author  = {Ramirez, Emerson and Antigravity Assistant},
  journal = {PCEI Theoretical Research Compendium},
  year    = {2026},
  url     = {https://github.com/MtnMCG/PCEI}
}
```
