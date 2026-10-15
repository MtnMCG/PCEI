# PCEI 455M — Complete Command Reference

## Architecture

| Stack | Layers | Dimensions | Role |
|---|---|---|---|
| **TextStack** | **16** | 1024D | Deep reasoning cortex |
| **ImageStack** | 5 | 640D | Visual pattern recognition |
| **AudioStack** | 4 | 320D | Temporal waveform processing |

**Total Parameters:** 455,048,574 (~455M)
**Checkpoint:** `checkpoints/pcei_500m.pt`
**Foundation:** `pcei_micro.pt` (15M, 128D, week-trained) center-planted into the 1024D weight space

All scripts automatically call `caffeinate` on macOS — your machine will stay awake for the full run.

---

## Setup

```bash
cd "/Users/emerson/Desktop/PCEI /pcei_framework"
source venv/bin/activate
```

---

## 1. Resonance Chat

Interactive conversation. Every message is a live training step.

```bash
python tools/resonance_chat.py --checkpoint-name pcei_500m.pt
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load and save to |
| `--depth` | `12` | Thinking depth (diffusion steps) — set higher for more stable reasoning |
| `--vocab-size` | `50257` | Vocabulary size (don't change) |

---

## 2. Dreaming Chamber

Internal self-consolidation. The model dreams on its own patterns with no external input.

```bash
python tools/dreaming_chamber.py --checkpoint-name pcei_500m.pt
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load/save |
| `--lr` | `1e-5` | Learning rate for dream training |
| `--save-interval` | `50` | Save every N dream cycles |

At the prompt, type a seed topic to prime the dreams, or press Enter for pure self-resonance.

---

## 3. Wikipedia Explorer

Autonomous Wikipedia crawler. Feeds articles via WitnessMode. Runs indefinitely until ctrl-c.

```bash
# Interactive topic selection
python tools/wiki_explorer.py --checkpoint-name pcei_500m.pt

# Skip the prompt, start on a topic
python tools/wiki_explorer.py --checkpoint-name pcei_500m.pt --start-topic "quantum physics"

# Flush confidence after migration for faster learning
python tools/wiki_explorer.py --checkpoint-name pcei_500m.pt --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load/save |
| `--start-topic` | `"Apple M4"` | Starting Wikipedia topic (skips prompt) |
| `--metabolic-flush` | off | Decays slow_field_confidence to absorb faster — use after migration |

---

## 4. Multimodal Tuner

Trains on paired image+text data. This is where cross-modal bonds form between the stacks.

```bash
python tools/multimodal_tuner.py --checkpoint-name pcei_500m.pt --steps 500

# With metabolic loss for stronger bond formation
python tools/multimodal_tuner.py --checkpoint-name pcei_500m.pt --steps 1000 --lr 1e-4 --use-metabolic-loss
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load/save |
| `--steps` | code default | Number of training steps |
| `--batch-size` | code default | Batch size per step |
| `--lr` | code default | Learning rate |
| `--grad-accum-steps` | code default | Gradient accumulation (effective batch = batch_size × this) |
| `--use-metabolic-loss` | off | Adds TopologyShifter proximity reward — strengthens cross-modal bonds |
| `--audio-only` | off | Skip images, only train text-to-audio resonance |
| `--image-only` | off | Skip audio, only train text-to-image resonance |

---

## 5. Instruction Tuner

Generates instruction data via Ollama and trains on it. Solidifies checkpoints every 50 steps and performs an emergency save on ctrl-c.
Requires Ollama running (`ollama serve`).

```bash
# Interactive mode menu (recommended — shows Normal/Basic/Chat selection)
python tools/instruction_tuner.py --checkpoint-name pcei_500m.pt

# Skip menu — go straight to a mode
python tools/instruction_tuner.py --mode auto --checkpoint-name pcei_500m.pt
python tools/instruction_tuner.py --mode basic --checkpoint-name pcei_500m.pt
python tools/instruction_tuner.py --mode chat --checkpoint-name pcei_500m.pt

# With options
python tools/instruction_tuner.py --mode auto --checkpoint-name pcei_500m.pt \
  --ollama-model llama3 --num-ollama 500 --metabolic-flush
```

| Flag | Default | What it does |
|---|---|---|
| `--mode` | interactive menu | `auto` = broad curriculum, `basic` = conversational only, `chat` = interactive |
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load/save |
| `--ollama-model` | `gpt-oss:120b-cloud` | Which Ollama model generates training data |
| `--num-ollama` | `100` | Number of examples to generate |
| `--force-generate` | off | Regenerate data even if cache exists |
| `--metabolic-flush` | off | Decay confidence for faster learning (use after migration) |
| **Depth-Stretching** | **Auto** | Trains with variable 8-12 diffusion steps to teach "Thinking Time" |
| **Safe-Save** | **Internal** | Catches `ctrl-c` to solidify the physics state before exiting |

### Mode Details

| Mode | Topics | Response Length | Goal |
|---|---|---|---|
| **Normal** | Math, history, philosophy, science, social | 256 tokens | General knowledge & reasoning |
| **Basic** | Greetings, small talk, farewells, reactions, identity | 80 tokens | How to talk to a person |
| **Chat** | You provide prompts interactively | — | Direct RLHF |

---

## 6. ChatML Data Importer

Converts standard third-party transformer training datasets (in ChatML format) into the native PCEI synthetic format, allowing PCEI to learn from advanced tool-use and reasoning datasets.

```bash
python tools/import_chatml.py ../hk_lora_chatml_cleaned_2026-06-05.jsonl
```

| Flag | Default | What it does |
|---|---|---|
| `input_file` | required | Path to your `.jsonl` ChatML file. |
| `--output` | `data/synthetic/instruction_tuning.jsonl` | Where to append the converted data. |

---

## 7. Full Curriculum Training

Three-phase training: Bootstrap (unimodal text) → Integration (cross-modal) → Open Learning.

```bash
# Standard run
python scripts/train.py --checkpoint-name pcei_500m.pt

# Long overnight run
python scripts/train.py --checkpoint-name pcei_500m.pt \
  --bootstrap-steps 2000 --integration-steps 3000 --open-steps 5000 \
  --lr 3e-4 --grad-accum-steps 4 --grad-clip 0.5

# Resume from where you left off
python scripts/train.py --checkpoint-name pcei_500m.pt --resume

# Synthetic data only
python scripts/train.py --checkpoint-name pcei_500m.pt --pure-synthetic

# PRESTIGE RUN: High Quality Only (Recommended for M4 Air)
python scripts/train.py \
  --checkpoint-name pcei_500m.pt \
  --resume \
  --high-quality-only \
  --lr 2e-4 \
  --grad-accum-steps 8 \
  --bootstrap-steps 500 \
  --integration-steps 500 \
  --open-steps 1000
```

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-name` | `pcei_micro.pt` | Checkpoint to load/save |
| `--lr` | `3e-4` | Base learning rate |
| `--batch-size` | `4` | Batch size |
| `--diffusion-steps` | code default | Denoising timesteps per forward pass |
| `--bootstrap-steps` | code default | Phase 1: unimodal text steps |
| `--integration-steps` | code default | Phase 2: cross-modal paired data steps |
| `--open-steps` | code default | Phase 3: open learning steps |
| `--log-every` | code default | Log metrics every N steps |
| `--eval-every` | code default | Run evaluation every N steps |
| `--plot-every` | code default | Generate plots every N steps |
| `--resume` | off | Resume from existing checkpoint step counter |
| `--pure-synthetic` | off | Only train on synthetic data |
| `--high-quality-only` | off | Only train on ShareGPT and Python Code (Prestige Diet). Uses optimized 2-step depth. |
| `--grad-accum-steps` | code default | Gradient accumulation steps |
| `--grad-clip` | code default | Gradient clipping max norm |

---

## Recommended Boot Sequence After Migration

```bash
# 1. Stabilize expanded weights + form cross-modal bonds
python tools/multimodal_tuner.py --checkpoint-name pcei_500m.pt --steps 500

# 2. Teach her to talk like a person first
python tools/instruction_tuner.py --mode basic --checkpoint-name pcei_500m.pt --metabolic-flush

# 3. Feed knowledge
python tools/wiki_explorer.py --checkpoint-name pcei_500m.pt --metabolic-flush

# 4. Broad instruction tuning
python tools/instruction_tuner.py --mode auto --checkpoint-name pcei_500m.pt

# 5. Dream to consolidate
python tools/dreaming_chamber.py --checkpoint-name pcei_500m.pt

# 6. Chat
python tools/resonance_chat.py --checkpoint-name pcei_500m.pt
# 7. M4 Air Prestige Config (Optimal)

For the 500M model on M4 hardware, use this sequence for maximum stability and logical depth:

```bash
# 1. THE PRESTIGE RUN (High-Quality Logic + Code)
python scripts/train.py --checkpoint-name pcei_500m.pt --resume --high-quality-only --lr 2e-4 --grad-accum-steps 8

# 2. DEPTH STRETCHING (Teach the model to 'think' for 20 steps)
python tools/instruction_tuner.py --mode auto --checkpoint-name pcei_500m.pt

# 3. INTERACTIVE RESONANCE (Test the results)
python tools/resonance_chat.py --checkpoint-name pcei_500m.pt --depth 12
```

---

## Technical Notes
- **MPS Optimization**: All tools use native 32-bit arithmetic. `autocast` is bypassed to prevent M4 Air driver hangs.
- **Atomic Snapshots**: Checkpoints are saved as `.tmp` first, then moved to `.pt` to prevent corruption during power loss or interrupts.

---

## Diagnostics

```bash
# Check checkpoint for NaN/corruption
python tools/numerical_audit.py

# Plot training metrics (replace <log_dir> with your actual metrics log folder, e.g. logs/20260520_120000)
python scripts/plot_metrics.py logs/<log_dir>

# Topology migration (scaling to a larger architecture)
python scripts/topology_migration.py --help
```
