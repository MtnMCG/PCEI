# PCEI 750M — Causal Autoregressive Command Reference

This document maps all execution setups, scripts, and flags updated for the **Causal Autoregressive SSM Pivot** and **Stateful Latent Pondering** on the 750M architecture.

---

## 1. Setup

Navigate to your workspace and activate the environment:
```bash
cd "/Users/emerson/Desktop/PCEI /pcei_framework"
source venv/bin/activate
```

---

## 2. Resonance Chat (Interactive Inference)

Runs interactive inference with **Latent Pondering** (Thought Recycling) and fast **$O(1)$ SSM State Caching**.

```bash
# Chat with metabolic flush (resets slow-field confidence to zero before conversation)
python tools/resonance_chat.py --checkpoint-name pcei_750m.pt --depth 12 --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load and save to. |
| `--depth` | `12` | **Max Latent Pondering Steps** — The maximum number of recycling cycles the model runs to settle its state before generating the first token. |
| `--metabolic-flush` | off | Resets slow-field confidence to zero before starting, letting weights adapt faster. |
| `--vocab-size` | `50257` | Vocabulary size. |

---

## 3. Dreaming Chamber (Self-Consolidation)

Allows the model to consolidate continuous patterns into its causal text weights. Automatically leverages **Dynamic Target Token Reconstruction** to map continuous memories into discrete tokens.

```bash
# Dream with metabolic flush
python tools/dreaming_chamber.py --checkpoint-name pcei_750m.pt --save-interval 50 --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load and save to. |
| `--lr` | `1e-5` | Learning rate. |
| `--save-interval` | `50` | Save every N dream cycles. |
| `--metabolic-flush` | off | Resets slow-field confidence to zero before starting, letting weights adapt faster. |

---

## 4. Wikipedia Explorer (Knowledge Crawling)

Crawls Wikipedia and trains the causal text stack on next-token prediction.

```bash
# Start on quantum physics with metabolic flush
python tools/wiki_explorer.py --checkpoint-name pcei_750m.pt --start-topic "quantum physics" --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load and save to. |
| `--start-topic` | `"quantum physics"` | Starting topic to seed crawler. |
| `--metabolic-flush` | off | Decays slow_field_confidence for faster learning (use after weight expansions). |

---

## 5. Instruction Tuner (RLHF, Alignment & HK Trading)

Trains the model on synthesized instruction-following tasks or specialized trading behavioral datasets.

```bash
# Run broad curriculum SFT with metabolic flush
python tools/instruction_tuner.py --mode auto --checkpoint-name pcei_750m.pt --metabolic-flush

# Run specialized HK Trading Mode SFT with metabolic flush
python tools/instruction_tuner.py --mode hk --checkpoint-name pcei_750m.pt --num-ollama 167 --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--mode` | interactive menu | `auto` (broad curriculum), `basic` (greeting/small talk), `chat` (interactive RLHF), `hk` (specialized trading behavior dataset). |
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load and save to. |
| `--ollama-model` | `gpt-oss:120b-cloud` | The LLM used to generate synthetic training text (for non-HK modes). |
| `--num-ollama` | `100` | Number of instructions to train on. |
| `--force-generate` | off | Re-generate text even if cache exists. |
| `--metabolic-flush` | off | Decay confidence for faster learning (use after migration). |

### SFT Mode Details

| Mode | Target Source | Goal |
|---|---|---|
| **auto** | Synthetic Ollama | General knowledge, math, history, philosophy, science |
| **basic** | Conversational synthetic | Conversational logic, greetings, identity alignment |
| **chat** | Interactive RLHF | Real-time direct feedback from prompt input |
| **hk** | `hk_lora_chatml_cleaned` | Specialized financial trading dossiers & tool execution behaviors |

---

## 6. Multimodal Tuner (Bond Formation)

Aligns text, image, and audio stacks. Forms and updates cross-modal physical bonds between stacks.

```bash
# Train cross-modal bonds with metabolic flush
python tools/multimodal_tuner.py --checkpoint-name pcei_750m.pt --use-metabolic-loss --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load and save to. |
| `--steps` | default | Training step count. |
| `--use-metabolic-loss` | off | Adds TopologyShifter coordinates proximity loss, strengthening cross-modal bonds. |
| `--audio-only` | off | Only trains text-to-audio bonds. |
| `--image-only` | off | Only trains text-to-image bonds. |
| `--metabolic-flush` | off | Resets slow-field confidence to zero before starting, letting weights adapt faster. |

---

## 7. Full Curriculum Training (Phase 1-3)

Trains from scratch or resumes across three phases (Bootstrap -> Integration -> Open Learning).

```bash
# High-quality prestige training run with metabolic flush
python scripts/train.py --checkpoint-name pcei_750m.pt --resume --high-quality-only --lr 2e-4 --grad-accum-steps 8 --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_750m.pt` | Checkpoint to load/save. |
| `--resume` | off | Resume training step counter from checkpoint. |
| `--high-quality-only` | off | Only train on high-quality text datasets (ShareGPT/Code). |
| `--lr` | `3e-4` | Base learning rate. |
| `--grad-accum-steps` | `4` | Gradient accumulation factor. |
| `--metabolic-flush` | off | Resets slow-field confidence to zero before starting, letting weights adapt faster. |

---

## Technical Notes

- **Causal Constraint**: The `TextStack` is now strictly causal. Bidirectional scanning is bypassed to prevent label leakage.
- **SSM Cache**: Generation utilizes the Mamba hidden state (`h`) and conv state (`conv_state`) to perform $O(1)$ updates per step.
- **Dynamic Decoders**: Training uses Shifted Cross-Entropy next-token prediction loss on text, while maintaining DDPM diffusion denoising on images and audio.
