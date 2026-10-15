================================================================================
                    PCEI FRAMEWORK — INTERACTION STANDARDS
                    Micro-Nexus Edition (16M Params, Ternary {0,1,2})
================================================================================

ENVIRONMENT SETUP (run once per terminal session):
  cd pcei_framework
  source venv/bin/activate
  export PYTHONPATH=$PYTHONPATH:.

CHECKPOINT: All tools share a single unified brain: checkpoints/pcei_micro.pt
SAFETY: Every save creates a backup (pcei_micro_BACKUP.pt). If a load fails, 
        the script aborts instead of overwriting.

================================================================================
 1. INITIALIZATION
================================================================================

[MICRO-NEXUS INITIALIZER]
File: tools/init_micro.py
Description: Creates a fresh 16M parameter blank-slate model with ternary 
{0,1,2} weights. Run this ONCE to create pcei_micro.pt. Running it again 
will overwrite the existing checkpoint.
Command:
  python3 tools/init_micro.py

================================================================================
 2. TRAINING TOOLS (tools/)
================================================================================

[WIKIPEDIA EXPLORER — Phase 1: Knowledge Ingestion]
File: tools/wiki_explorer.py
Description: Autonomous Wikipedia browsing. The model reads articles in 
Witness Mode, learning language patterns. When it gets "bored" (gradient 
norm drops), the DreamSeeder generates an internal thought to pick the 
next article. The model chooses its own curriculum.
Command:
  python3 tools/wiki_explorer.py
Arguments:
  --checkpoint-name   (default: "pcei_micro.pt")
  --start-topic       (default: "Apple M4")
  --d-model-text      (default: 128)
  --layers-text       (default: 3)

[MULTIMODAL TUNER — Phase 2: Sensory Grounding]
File: tools/multimodal_tuner.py
Description: Trains the Image and Audio stacks using your labeled multimodal 
dataset. Two modes: Captioning (image → text) and Generation (text → image).
Grounds vocabulary in real-world visual and auditory data.
Command:
  python3 tools/multimodal_tuner.py
Arguments:
  --checkpoint-name   (default: "pcei_micro.pt")
  --batch-size        (default: 4)
  --steps             (default: 1000)
  --lr                (default: 5e-5)
  --grad-accum-steps  (default: 4)
  --use-metabolic-loss (flag to enable physics-based reward shaping)

[DREAMING CHAMBER — Phase 3: Self-Resonance]
File: tools/dreaming_chamber.py
Description: Pure internal dialogue. No external data. The DreamSeeder 
generates noise shaped by the AmbientField, the model denoises it, and 
the result is compared against the slow_state (long-term memory). The 
model consolidates, reorganizes topology, and forms/dissolves entanglement 
bonds autonomously. You can optionally prime with a seed topic or let it 
dream from pure noise. Watch the physics engine live.
Command:
  python3 tools/dreaming_chamber.py
Arguments:
  --checkpoint-name   (default: "pcei_micro.pt")
  --lr                (default: 1e-5)
  --save-interval     (default: 50, save every N dream cycles)

[RESONANCE CHAT — Real-Time Interactive Learning]
File: tools/resonance_chat.py
Description: Talk to the model. Every message you type triggers a training 
step — the model learns your words in real-time. It responds, then runs 
one internal consolidation step. Watch the loss drop as it adapts to you.
Command:
  python3 tools/resonance_chat.py
Arguments:
  --checkpoint        (default: "checkpoints/pcei_micro.pt")

[INSTRUCTION TUNER — Behavioral Alignment]
File: tools/instruction_tuner.py
Description: Training modes:
  - "auto": Automated SFT using synthetic instruction pairs from Ollama
  - "basic": Strictly conversational training (greetings, small talk, etc.)
  - "hk": Exclusive training on the hk_lora_chatml_cleaned dataset for trading behaviors
  - "chat": Interactive RLHF where you rate responses 1-5 and the model
            updates via gradient descent (good) or ascent (bad)
Commands:
  python3 tools/instruction_tuner.py --mode chat   # Interactive RLHF
  python3 tools/instruction_tuner.py --mode auto   # Automated SFT
  python3 tools/instruction_tuner.py --mode hk     # HK Trading SFT
Arguments:
  --checkpoint-name   (default: "pcei_micro.pt")
  --ollama-model      (default: "gpt-oss:120b-cloud")
  --num-ollama        (default: 100)
  --force-generate    (flag to regenerate Ollama dataset)

================================================================================
 3. UTILITY TOOLS
================================================================================

[DATA HUB]
File: tools/data_hub.py
Description: Rich CLI dashboard to view and manage processed datasets.
Command:
  python3 tools/data_hub.py

[MULTIMODAL LABELER]
File: tools/multimodal_labeler.py
Description: Interactive tool to map Image/Video files to text descriptions,
generating training_database.jsonl.
Command:
  python3 tools/multimodal_labeler.py

================================================================================
 4. LEGACY TRAINING (scripts/)
================================================================================

[MAIN TRAINER — Full Curriculum (Legacy)]
File: scripts/train.py
Description: The original Phase 1/2/3 curriculum trainer. Now with automatic 
checkpoint loading (no --resume flag needed). Supports bootstrap, integration, 
and open learning phases in sequence.
Command:
  python3 scripts/train.py [options]
Arguments:
  --lr                (default: 3e-4)
  --batch-size        (default: 4)
  --grad-accum-steps  (default: 4)
  --bootstrap-steps   (default: 250)
  --integration-steps (default: 250)
  --open-steps        (default: 0)
  --checkpoint-name   (default: "pcei_checkpoint.pt")

================================================================================
 5. INFERENCE & VERIFICATION
================================================================================

[COMPREHENSIVE INFERENCE TEST]
File: scripts/comprehensive_inference_test.py
Command:
  python3 scripts/comprehensive_inference_test.py --checkpoint pcei_micro.pt

[VISUAL INFERENCE TEST]
File: scripts/visual_inference_test.py
Command:
  python3 scripts/visual_inference_test.py

[PLOT METRICS]
File: scripts/plot_metrics.py
Command:
  python3 scripts/plot_metrics.py

================================================================================
 RECOMMENDED TRAINING SEQUENCE
================================================================================

 1. Initialize:          python3 tools/init_micro.py
 2. Feed Knowledge:      python3 tools/wiki_explorer.py
 3. Ground Senses:       python3 tools/multimodal_tuner.py
 4. Let It Dream:        python3 tools/dreaming_chamber.py
 5. Talk To It:          python3 tools/resonance_chat.py
 6. Align Behavior:      python3 tools/instruction_tuner.py --mode chat

 All tools share pcei_micro.pt. Knowledge compounds across modalities.
 Run them in any order after Step 1. Each session builds on the last.

================================================================================
 ARCHITECTURE NOTES
================================================================================

 - Ternary Weights: {0, 1, 2} only. No negative values. Metabolic purity.
 - Mamba SSM: No attention anywhere. Linear-time sequence processing.
 - Physical Constraints: Interaction cost scales with spatial distance.
 - Dual-Timescale Memory: AmbientField fast (working) + slow (consolidated).
 - Always-On Learning: Nothing is ever frozen. Physics prevents forgetting.
 - 16M Parameters: Micro-Nexus (128-wide text/img, 64 audio, 3/2/1 layers).

================================================================================
