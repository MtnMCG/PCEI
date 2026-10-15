# PCEI 750M: End-to-End Curriculum & Data Pipeline Specification

**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2  
**Target Checkpoint**: `pcei_750m_reasoning_assistant_latest.pt` (846M Parameters)  
**Scope**: Complete Cognitive Lifecycle (Phase 1 Pretraining $\to$ Phase 2 Dreaming $\to$ Phase 3 Blended Instruction)  
**Total Tokens Trained**: $333.8\text{M}$ Pretraining Tokens + Active Multi-Turn Instruction Stream at $L=2048$  
**Date**: September 2026  

---

## Executive Overview: The Three-Phase Curriculum Architecture

The PCEI framework rejects the brute-force monolithic pretraining paradigm. In standard LLMs, models are trained uniformly on trillions of chaotic web tokens with zero structural or cognitive staging. In contrast, PCEI structures learning into **three distinct biological training stages**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: FOUNDATION PRETRAINING (PHASE 1)                 │
│  Volume: 333,824,000 Tokens | 163,000 Steps | Context: L = 512              │
│  Data Mixture: Literature, Wikipedia, Algorithmic Synthetics, Multimodal    │
│  Objective: Establish syntactic basis, algorithmic grokking, representation│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 STAGE 2: AUTONOMOUS DREAMING REPLAY (PHASE 2)               │
│  Volume: 1,000 Closed-Loop Denoising Cycles | 0 External Tokens Ingested    │
│  Objective: Generative replay in continuous latent space, manifold          │
│             stabilization, PPL contraction (57.79 → 38.72)                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             STAGE 3: BLENDED MULTI-TURN INSTRUCTION TUNING (PHASE 3)         │
│  Volume: 150,000 Production Steps | Context: L = 2048 | 100% Sequence Pack  │
│  Data Mixture: 42% Claude/RLHF, 20% ShareGPT, 18% UltraChat, 10% Teacher    │
│  Objective: Formal reasoning, multi-turn dialogue, tool & instruction adherence │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part I: Foundation Pretraining Curriculum (Phase 1)

Phase 1 established the grammatical, syntactic, mathematical, and cross-modal bedrock of the 846M-parameter foundation model before any instruction tuning took place.

### 1. Pretraining Data Corpora & Mixture Composition

Pretraining spanned **163,000 optimization steps** at context length $L = 512$ with micro-batch 1 and gradient accumulation 4 ($2,048\text{ tokens/step}$), totaling **$333,824,000\text{ tokens}$**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1 PRETRAINING CURRICULUM MIXTURE                    │
├───────────────────────────────┬─────────────┬───────────────────────────────┤
│ Data Corpus & Source          │ Share (%)   │ Primary Function / Objective  │
├───────────────────────────────┼─────────────┼───────────────────────────────┤
│ 1. Classical Literature       │    25.0%    │ Complex sentence syntax, rich │
│    (Project Gutenberg)        │             │ vocabulary, long-range prose  │
│ 2. Wikipedia Knowledge Base   │    25.0%    │ Factual encyclopedic world    │
│    (`wiki_train`, `wiki_test`)│             │ knowledge, scientific context │
│ 3. Algorithmic Synthetics     │    20.0%    │ Group operations, modular math│
│    (Modular math, induction)  │             │ grokking, associative recall  │
│ 4. Foundation Teacher Data    │    15.0%    │ High-density synthetic text   │
│    (`pcei_foundation_teacher`)│             │ across logic, code, & physics │
│ 5. Modern Complex Literature  │    10.0%    │ Non-linear narratives, dense  │
│    (Dune, Infinite Jest, etc.)│             │ vocabulary, multi-character   │
│ 6. Multimodal Concept Ground  │     5.0%    │ MS-COCO image-caption &       │
│    (COCO & LibriSpeech pairs) │             │ LibriSpeech audio alignment   │
└───────────────────────────────┴─────────────┴───────────────────────────────┘
```

### 2. Deep Dive: Pretraining Corpus Tiers

#### 2.1 Classical & Scientific Literature (Project Gutenberg) — 25.0%
* **Location**: `data/raw/text/gutenberg/`
* **Contents**: Over 2,000 public-domain classical texts covering natural philosophy, mathematics, history, and classical literature.
* **Role**: Teaches the 24-layer Mamba TextStack natural syntactic parsing, grammatically complex subordinate clauses, and rich lexical variation, preventing degenerate repetition loops.

#### 2.2 Encyclopedic World Knowledge (Wikipedia) — 25.0%
* **Location**: `data/raw/text/wiki_train.txt`, `wiki_test.txt`, `wiki_valid.txt`
* **Contents**: Millions of words of factual, structured encyclopedic articles.
* **Role**: Establishes broad factual grounding across biology, chemistry, physics, computer science, geography, and world history.

#### 2.3 Algorithmic Synthetics & Grokking Curricula — 20.0%
* **Location**: `data/synthetic/`
  * `modular_arithmetic.jsonl`: Mathematical group operations $(a \circ b) \pmod p$ for prime $p=97, 113$. Designed specifically to detect mathematical "grokking" (sudden phase transitions from memorization to generalization).
  * `sequence_patterns.jsonl`: Synthetic symbol induction heads, pattern completion, and associative recall tasks.
  * `congruent_concepts.jsonl`: Hierarchical concept trees and ontological relationships ("A mammal is an animal", "A dog is a mammal").
  * `beast_reasoning.jsonl`: High-difficulty synthetic deductive logic syllogisms.

#### 2.4 Foundation Teacher Distillation — 15.0%
* **Location**: `data/curriculum/pcei_foundation_teacher.jsonl` (4.68 MB)
* **Contents**: High-density synthetic curriculum generated by teacher models spanning algorithmic coding, formal mathematical proofs, and physical dynamics.

#### 2.5 Modern Dense Literature — 10.0%
* **Location**: `data/raw/text/modern_books/` & `unprocessed books/`
* **Contents**: Challenging linguistic masterworks: Frank Herbert's *Dune*, David Foster Wallace's *Infinite Jest*, Cormac McCarthy's *Blood Meridian*, and Thomas Pynchon's *Gravity's Rainbow*.
* **Role**: Exposes the selective state transitions to high linguistic entropy, dense vocabulary, non-standard syntax, and shifting conversational perspectives.

#### 2.6 Multimodal Sensory Grounding — 5.0%
* **Location**: `data/raw/images/`, `data/raw/audio/`, `coco_ann.zip`, `librispeech_dev.tar.gz`
* **Contents**: Paired MS-COCO image-caption pairs and LibriSpeech audio-transcription pairs.
* **Role**: Trains the boundary layers ($\mathbf{W}_{\text{img}\to\text{text}}$ and $\mathbf{W}_{\text{audio}\to\text{img}}$) to bind sensory embeddings to text semantics under Scheduled Sensory Dropout ($\mathcal{P}_{\text{drop}} = 0.30$).

### 3. Pretraining Optimization Hyperparameters (Phase 1)
* **Context Length ($L$)**: `512 tokens`
* **Learning Rate Schedule**: **Warmup-Stable-Decay (WSD)**:
  * Warmup: First $5\%$ of steps ($0 \to 3.0 \times 10^{-4}$)
  * Stable: $80\%$ of steps at constant $\eta = 3.0 \times 10^{-4}$
  * Cosine Decay: Final $15\%$ of steps decaying to $\eta_{\text{min}} = 3.0 \times 10^{-5}$
* **Weight Quantization**: BitNet 1.58b dynamic threshold scaling ($\gamma = \frac{1}{mn}\sum |W_{ij}|$) on all TextStack feed-forward linear layers.
* **Pretraining Result**: Validation loss decreased from $10.82 \to 3.65$ nats; held-out perplexity reached **$38.72$** on merely $1.97\%$ of Chinchilla optimal token allocation.

---

## Part II: Autonomous Dreaming Consolidation (Phase 2)

Before proceeding to instruction tuning, the model underwent **Phase 2: The Dreaming Chamber** ([`tools/dreaming_chamber.py`](file:///Users/emerson/Desktop/PCEI%20/pcei_framework/tools/dreaming_chamber.py)):
* **Input Tokens**: **Zero external tokens**.
* **Duration**: 1,000 autonomous closed-loop cycles.
* **Mechanism**: Continuous latent diffusion in embedding space using `DreamSeeder` directed noise.
* **Result**:
  * Denoise loss converged from $0.4289 \to 0.0000$.
  * Held-out text perplexity contracted from **$57.79 \to 38.72$**.
  * Dynamic entanglement bonds stabilized between $136$ and $159$ active links.
  * Plastic confidence calcified to $\bar{\sigma}_{\text{conf}} = 0.2500$, locking in foundation representations.

---

## Part III: Blended Multi-Turn Instruction Curriculum (Phase 3)

Phase 3 transitions the consolidated foundation model into a production conversational and reasoning assistant operating at context length $L = 2048$.

### 1. Production Curriculum Probability Distribution

The `BlendedDatasetStreamer` concurrently streams from high-density instruction and relational anchoring datasets with the following calibrated probabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             PHASE 3 RELATIONAL-ANCHORED INSTRUCTION CURRICULUM MIXTURE      │
├────────────────────────┬─────────────┬──────────────────────────────────────┤
│ Curriculum Source      │ Probability │ Primary Cognitive Function           │
├────────────────────────┼─────────────┼──────────────────────────────────────┤
│ 1. Claude-CoT          │    25.0%    │ Step-by-step chain-of-thought & math │
│ 2. ShareGPT Multi-Turn │    15.0%    │ Extended dialogue context following  │
│ 3. Beast-Deductive-Log │    12.0%    │ Strict multi-premise syllogisms      │
│ 4. Congruent-Concepts  │    10.0%    │ Entity-attribute relational anchor   │
│ 5. Sequence-Patterns   │     5.0%    │ Inductive pattern & symbol recall    │
│ 6. Anthropic-RLHF      │    10.0%    │ Preference adherence & helpfulness   │
│ 7. UltraChat-200k      │     8.0%    │ Open-domain conversational breadth   │
│ 8. Foundation Teacher  │    10.0%    │ Ground truth reasoning distillation  │
│ 9. FineWeb-Edu         │    10.0%    │ STEM factual & academic anchor       │
└────────────────────────┴─────────────┴──────────────────────────────────────┘
   Total Relational & Deductive Anchor: 27.0% (Topic Drift & Semantic Lock)
```

### 2. Tier Details

#### 2.1 Claude Chain-of-Thought (Claude-CoT) — 25.0%
* **Source**: Curated Claude 3.5 Sonnet / Opus synthetic reasoning traces (`data/synthetic/anthropic_claude.jsonl`, 53 MB).
* **Format**: Structured step-by-step deductive reasoning paths.
* **Function**: Inoculates the Mamba TextStack against impulsive next-token guessing, teaching the recurrent state to ponder and execute intermediate reasoning steps before emitting answers.

#### 2.2 Relational & Deductive Anchors (Beast + Congruent + Sequence) — 27.0%
* **Sources**: `data/synthetic/beast_reasoning.jsonl`, `congruent_concepts.jsonl`, `sequence_patterns.jsonl`.
* **Function**: **Prevents Semantic Random Walks & Hallucinatory Topic Drift.** Enforces strict subject-predicate invariants in the Mamba hidden state ($h_t$). Ensures that when the user asks about entity $A$, the state representation remains bound to $A$'s properties rather than drifting into ungrounded conversational associations.

#### 2.3 ShareGPT Multi-Turn Conversations — 15.0%
* **Source**: Real-world multi-turn dialogues (`data/high_quality/sharegpt_logic.jsonl`).
* **Function**: Trains the recurrent state transitions to track conversational threads across 3 to 12 turns without hallucination.

#### 2.4 UltraChat-200k — 18.0%
* **Source**: High-diversity open-domain conversational dataset.
* **Function**: Broadens everyday lexical fluency, idioms, and casual dialogue handling.

#### 2.5 Council-Ultima Synthetic Teacher — 10.0%
* **Source**: Automated local teacher pipeline querying `MtnMCG/Council-Ultima:latest` on Ollama via [`tools/teacher_generator.py`](file:///Users/emerson/Desktop/PCEI%20/pcei_framework/tools/teacher_generator.py).
* **Function**: Continuously feeds customized, high-density reasoning, logic puzzles, algorithmic coding, and math demonstrations into `data/curriculum/council_teacher.jsonl`.

#### 2.6 FineWeb-Edu (Factual Foundation Anchor) — 10.0%
* **Source**: Curated educational web text with high textbook-quality scoring.
* **Function**: Acts as a persistent pretraining anchor, mitigating catastrophic forgetting of scientific, historical, and factual knowledge during conversational tuning.

---

## Part IV: Continuous Sequence Packing Protocol (Token-Maxx)

In standard training pipelines, variable-length documents are padded with zeros to fixed sequence lengths:
$$\text{Standard: } [\mathbf{x}_1, \dots, \mathbf{x}_{600}, \underbrace{0, 0, \dots, 0}_{1448 \text{ padding zeros}}]$$
* This wastes $35-50\%$ of GPU memory and compute on non-informative null tokens.

### The Token-Maxx Continuous Packing Solution
PCEI implements continuous sequence packing in both [`scripts/train.py`](file:///Users/emerson/Desktop/PCEI%20/pcei_framework/scripts/train.py) and [`tools/blended_instruction_tuner.py`](file:///Users/emerson/Desktop/PCEI%20/pcei_framework/tools/blended_instruction_tuner.py):
1. Documents from all active curriculum streams are tokenized into BPE IDs.
2. An `<|endoftext|>` token (ID `50256`) is appended to the conclusion of each document.
3. The streamer fills an active FIFO token queue:
   $$\mathcal{Q} = [\text{Doc}_1, \langle|\text{eot}|\rangle, \text{Doc}_2, \langle|\text{eot}|\rangle, \dots, \text{Doc}_N]$$
4. Exactly $L = 2048$ tokens are sliced for each micro-batch:
   $$\mathbf{X} = \mathcal{Q}[:2048], \quad \mathbf{Y} = \mathcal{Q}[1:2049]$$
* **Result**: **$100.0\%$ active compute efficiency**. Zero FLOPs are wasted on padding. This enables sustained throughput of **$1,475\text{ tokens/second}$** on a single NVIDIA RTX 4090.

---

## Part V: Data Directory Mapping & Ingestion Mechanics

The `src/core/ingestors.py` module and `load_text_data()` dynamically parse diverse dataset schemas without code modification:

```
pcei_framework/data/
├── curriculum/
│   ├── council_teacher.jsonl          # Ollama Council-Ultima live teacher samples
│   └── pcei_foundation_teacher.jsonl  # Phase 1 foundation teacher curriculum (4.68MB)
├── high_quality/
│   ├── python_structure.jsonl         # Python AST code logic
│   └── sharegpt_logic.jsonl           # Multi-turn conversational reasoning
├── synthetic/
│   ├── anthropic_claude.jsonl         # Claude Chain-of-Thought dataset (53.4MB)
│   ├── beast_reasoning.jsonl          # High-difficulty deductive logic syllogisms
│   ├── congruent_concepts.jsonl       # Semantic hierarchy & ontological trees
│   ├── hk_expanded.jsonl              # Domain reasoning & system knowledge
│   ├── instruction_tuning.jsonl       # Instruction-response dialogue pairs
│   ├── modular_arithmetic.jsonl       # Mathematical group operations (Z/pZ)
│   ├── prompt_formatting.jsonl        # ChatML & dialogue formatting
│   └── sequence_patterns.jsonl        # Induction heads & pattern completion
├── raw/
│   ├── text/
│   │   ├── gutenberg/                 # Public domain classical literature
│   │   ├── modern_books/              # High-entropy modern literature
│   │   ├── mass_reasoning/            # Mathematical & algorithmic text
│   │   ├── wiki_train.txt             # Wikipedia pretraining text
│   │   ├── wiki_valid.txt             # Held-out validation corpus
│   │   └── wiki_test.txt              # Test perplexity benchmark corpus
│   ├── images/ & coco_ann.zip         # MS-COCO image-caption pairs
│   └── audio/ & librispeech_dev.tar.gz# LibriSpeech audio-transcription pairs
```

### Supported Record Formats in Ingestion Pipeline:
1. Direct Text: `{"text": "..."}`
2. Prompt / Response: `{"prompt": "...", "response": "..."}`
3. Alpaca / CodeFeedback: `{"instruction": "...", "input": "...", "output": "..."}`
4. OpenAI Messages: `{"messages": [{"role": "user", "content": "..."}, ...]}`
5. ShareGPT Conversations: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
