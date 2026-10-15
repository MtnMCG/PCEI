# PCEI 750M: User Operations & Testing Runbook

**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt` (846M Parameters)  
**Primary Execution Environments**: Apple Silicon (Mac M-Series Unified Memory) & NVIDIA RTX 4090 Cloud  
**Date**: September 2026  

---

## 1. Local Environment Setup

Open a terminal on your Mac and navigate to the project directory:

```bash
cd "pcei_framework"
source venv/bin/activate
```

*Verify your environment:*
```bash
python3 -c "import torch; print('PyTorch Version:', torch.__version__, '| MPS Available:', torch.backends.mps.is_available())"
```

---

## 2. Syncing Latest Model Checkpoints From Cloud

The 2048-token instruction tuner is actively running on the Vast.ai RTX 4090 cloud server and periodically saving checkpoints. To sync the newest weights down to your local machine:

```bash
# Execute the automated bidirectional sync script:
./sync_from_cloud.sh
```

Or manually pull the latest checkpoint via `rsync`:
```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p <SSH_PORT> -o StrictHostKeyChecking=no" \
  root@<REMOTE_GPU_IP>:/workspace/pcei/checkpoints/pcei_750m_reasoning_assistant_latest.pt \
  checkpoints/
```

---

## 3. Interactive Resonance Chat (Local Testing)

Run interactive conversational inference using the 24-layer Mamba TextStack with latent pondering and continuous ambient field modulation.

### Basic Launch (Apple Silicon MPS / Local GPU):
```bash
python3 tools/resonance_chat.py \
  --checkpoint-name pcei_750m_reasoning_assistant_latest.pt \
  --device mps \
  --max-context 2048 \
  --temperature 0.7
```

### Advanced Chat Options:
```bash
python3 tools/resonance_chat.py \
  --checkpoint-name pcei_750m_reasoning_assistant_latest.pt \
  --device mps \
  --depth 12 \
  --metabolic-flush \
  --top-p 0.9 \
  --max-context 2048
```

| Parameter | Default | Function |
| :--- | :--- | :--- |
| `--checkpoint-name` | `pcei_750m_reasoning_assistant_latest.pt` | Specifies the weights file to load from `checkpoints/`. |
| `--device` | `mps` (or `cpu`) | Target backend on Mac (`mps` utilizes Apple Silicon Unified Memory GPU). |
| `--depth` | `12` | Max latent pondering iterations before emitting the first token. |
| `--metabolic-flush` | `off` | Resets slow-field confidence to 0 before dialogue to maximize responsiveness. |
| `--max-context` | `2048` | Sliding window token buffer for multi-turn history. |
| `--temperature` | `0.7` | Sampling temperature for next-token generation. |

---

## 4. Local Synthetic Teacher Generation (Ollama Council-Ultima)

PCEI includes an automated high-density synthetic reasoning distillation pipeline that queries `MtnMCG/Council-Ultima:latest` running locally on Ollama (`http://127.0.0.1:11434`).

### Ensure Ollama is Running:
```bash
ollama list
# If needed, pull the teacher model:
ollama pull MtnMCG/Council-Ultima:latest
```

### Generate Teacher Curriculum Samples:
```bash
# Generate 10 high-density reasoning, logic, and math pairs:
python3 tools/teacher_generator.py \
  --model "MtnMCG/Council-Ultima:latest" \
  --samples 10 \
  --output data/curriculum/council_teacher.jsonl
```

### Continuous Background Distillation:
```bash
# Run continuous distillation in the background:
nohup python3 tools/teacher_generator.py \
  --model "MtnMCG/Council-Ultima:latest" \
  --continuous \
  --interval 5 \
  --output data/curriculum/council_teacher.jsonl > logs/teacher_daemon.log 2>&1 &
```

---

## 5. Phase 2 Autonomous Dreaming Chamber (Local Sleep Consolidation)

To run offline latent diffusion sleep consolidation locally, contracting the manifold and stabilizing representations without ingesting any external text:

```bash
python3 tools/dreaming_chamber.py \
  --checkpoint-name pcei_750m_reasoning_assistant_latest.pt \
  --cycles 100 \
  --save-interval 25 \
  --device mps \
  --lr 1.0e-5
```

*What happens during dreaming:*
1. The model creates directed noise in its latent embedding space shaped by `AmbientField`.
2. Reverse diffusion denoises the trajectory, mapping continuous memories into discrete tokens.
3. Plastic confidence calcifies on recurring attractors, locking in stability.

---

## 6. Monitoring & Inspecting Cloud Production Training

The production instruction tuning run is hosted on an NVIDIA RTX 4090 on Vast.ai.

### Connect & Inspect Live Training:
```bash
ssh -i ~/.ssh/id_ed25519 -p <SSH_PORT> -o StrictHostKeyChecking=no root@<REMOTE_GPU_IP>
```

### Inside the Cloud Instance:
```bash
# 1. Check training process status:
ps aux | grep blended

# 2. View live training log (loss, PPL, tok/s, active bonds):
tail -f /workspace/blended_instruction_2048.log

# 3. Check GPU temperature and VRAM utilization:
nvidia-smi

# 4. Check Credit Guard status:
tail -f /workspace/credit_guard.log
```

---

## 7. Running Deterministic Scientific Diagnostics

To re-run the exact empirical benchmarks cited in the 6 scientific papers:

```bash
python3 tools/run_empirical_benchmarks.py \
  --checkpoint-name pcei_750m_reasoning_assistant_latest.pt \
  --output docs/sources/empirical_benchmark_results.json
```

This tests:
1. **Ambient Field Shannon Collapse**: Scalar mean-pooling information loss factor and token CosSim.
2. **Entanglement Bonds Gradient Neutrality**: Controlled float-exact backprop ablation ($176$ active bonds vs $0$ bonds).
3. **BitNet 1.58b Parameter Distribution**: Layer-by-layer audit across all 332.6M ternary weights.
