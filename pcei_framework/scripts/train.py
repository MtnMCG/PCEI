"""
PCEI Training Loop — Full Curriculum with Grokking Detection

Now with:
- Real structured synthetic data (modular arithmetic + patterns)
- Train/eval split for grokking detection
- 128-token context window
- Accuracy tracking on eval set
- All physics subsystems active

Usage:
    python scripts/train.py [--steps 500] [--lr 3e-4]
"""
import torch
import torch.nn as nn
import json
import argparse
import sys
import os
import random
import math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.pcei import PCEINexus
from src.engine.metrics import PCEIMetrics
from src.engine.diffusion_utils import diffusion_training_step
# Resonance/Topology now managed via nexus.core
from src.core.ingestors import TextIngestor, PCEIDataset
from src.modules.plasticity import PlasticLayer

try:
    from scripts.plot_metrics import generate_all_plots
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

import tiktoken

MAX_SEQ_LEN = 128  # Must match model's TEXT_SEQ_LEN — 512 at 455M causes MPS hangs


# ═══════════════════════════════════════
# DATA: BPE Tokenizer + Loader
# ═══════════════════════════════════════
class PCEITokenizer:
    """
    BPE tokenizer using tiktoken (GPT-2 encoding).
    Explicitly preserves <|endoftext|> (ID 50256) as the EOS/BOS signal.
    """
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256 # Standard tiktoken GPT-2 EOS
    
    def encode_raw(self, text: str) -> list:
        """Encodes text to a raw list of token IDs without padding or truncation."""
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        return [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        # tiktoken's gpt2 doesn't use 0, but we use it for PAD.
        # We'll map tiktoken IDs directly, but ensure they don't hit 0.
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        
        # Security check: ensure no 0 IDs (which we use for PAD)
        # Most tiktoken IDs are > 0, but if any are 0, we'll map them to a high range
        ids = [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]
        
        # Truncate
        ids = ids[:max_len]
        # Pad
        while len(ids) < max_len:
            ids.append(self.pad_id)
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        # Skip PAD (0)
        raw_ids = [tok_id.item() for tok_id in ids if tok_id.item() != self.pad_id]
        # Handle our 0-mapping if any
        raw_ids = [0 if tok_id == (self.vocab_size - 1) else tok_id for tok_id in raw_ids]
        
        try:
            return self.enc.decode(raw_ids)
        except Exception:
            return "<decode_error>"


class ContinuousSequencePacker:
    """
    Continuous Sequence Packer for optimal hardware utilization on Apple Silicon / MPS.
    Concatenates variable-length documents separated by <|endoftext|> (eos_id)
    into fixed-size windows of length max_seq_len without padding zeros.
    Guarantees 100% of tokens computed during training are real information tokens.
    """
    def __init__(self, texts: list, tokenizer: PCEITokenizer, max_seq_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer = []
        self.doc_indices = list(range(len(texts))) if texts else []
        self.doc_ptr = 0
        if self.doc_indices:
            random.shuffle(self.doc_indices)

    def _refill_buffer(self, min_tokens: int):
        if not self.texts:
            fallback = self.tokenizer.encode_raw("placeholder text")
            if not fallback or fallback[-1] != self.tokenizer.eos_id:
                fallback.append(self.tokenizer.eos_id)
            while len(self.buffer) < min_tokens:
                self.buffer.extend(fallback)
            return

        while len(self.buffer) < min_tokens:
            if self.doc_ptr >= len(self.doc_indices):
                self.doc_indices = list(range(len(self.texts)))
                random.shuffle(self.doc_indices)
                self.doc_ptr = 0

            idx = self.doc_indices[self.doc_ptr]
            self.doc_ptr += 1

            if idx < len(self.texts):
                doc = self.texts[idx]
                if isinstance(doc, str) and doc.strip():
                    safe_doc = doc[:10000]
                    tokens = self.tokenizer.encode_raw(safe_doc)
                    if tokens:
                        if tokens[-1] != self.tokenizer.eos_id:
                            tokens.append(self.tokenizer.eos_id)
                        self.buffer.extend(tokens)

    def get_batch(self, batch_size: int, device: str = "cpu") -> dict:
        total_tokens = batch_size * self.max_seq_len
        self._refill_buffer(total_tokens)

        batch_slice = self.buffer[:total_tokens]
        self.buffer = self.buffer[total_tokens:]

        token_tensor = torch.tensor(batch_slice, dtype=torch.long).view(batch_size, self.max_seq_len).to(device)
        return {
            'text': token_tensor,
            'image': None,
            'audio': None
        }


def extract_text_from_record(record: dict) -> str:
    """
    Extracts text from diverse JSONL record formats:
    - { "text": ... }
    - { "prompt": ..., "response": ... } (e.g. Anthropic Claude, hk_expanded, instruction_tuning)
    - { "instruction": ..., "output": ... } (with optional "input")
    - { "messages": [ {"role": ..., "content": ...}, ... ] }
    - { "conversations": [ {"from": ..., "value": ...}, ... ] }
    """
    if not isinstance(record, dict):
        return ""
    
    # 1. Direct text field
    text = record.get('text')
    if isinstance(text, str) and text.strip():
        return text.strip()
    
    # 2. Prompt / Response schema (e.g. anthropic_claude.jsonl, hk_expanded.jsonl, instruction_tuning.jsonl)
    prompt = record.get('prompt')
    response = record.get('response')
    if prompt is not None or response is not None:
        p_str = str(prompt).strip() if prompt else ""
        r_str = str(response).strip() if response else ""
        if p_str and r_str:
            if "ASSISTANT:" in p_str or "USER:" in p_str:
                return f"{p_str} {r_str}".strip()
            return f"USER: {p_str}\nASSISTANT: {r_str}".strip()
        elif p_str:
            return p_str
        elif r_str:
            return r_str
            
    # 3. Instruction / Output schema (ShareGPT / Alpaca / CodeFeedback)
    instruction = record.get('instruction')
    output = record.get('output')
    inp = record.get('input', '')
    if instruction is not None or output is not None:
        parts = []
        if instruction:
            parts.append(f"USER: {str(instruction).strip()}")
        if inp:
            parts.append(f"Input: {str(inp).strip()}")
        if output:
            parts.append(f"ASSISTANT: {str(output).strip()}")
        if parts:
            return "\n".join(parts)

    # 4. Chat messages schema: [{"role": "user", "content": ...}, ...]
    messages = record.get('messages')
    if isinstance(messages, list) and messages:
        msg_parts = []
        for m in messages:
            if isinstance(m, dict):
                r = m.get('role', 'user').lower()
                role_tag = "USER" if r in ['user', 'human'] else ("ASSISTANT" if r in ['assistant', 'bot', 'gpt'] else "SYSTEM")
                content = m.get('content', '')
                if content:
                    msg_parts.append(f"{role_tag}: {str(content).strip()}")
        if msg_parts:
            return "\n".join(msg_parts)

    # 5. ShareGPT conversations schema: [{"from": "human", "value": ...}, ...]
    conversations = record.get('conversations')
    if isinstance(conversations, list) and conversations:
        conv_parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                speaker = turn.get('from', 'human').lower()
                speaker_tag = "USER" if speaker in ['human', 'user'] else ("ASSISTANT" if speaker in ['gpt', 'chatgpt', 'assistant', 'bot'] else "SYSTEM")
                val = turn.get('value', '')
                if val:
                    conv_parts.append(f"{speaker_tag}: {str(val).strip()}")
        if conv_parts:
            return "\n".join(conv_parts)

    return ""


def load_text_data(
    synthetic_dir: Path = None,
    raw_text_dir: Path = None,
    pure_synthetic: bool = False,
    high_quality_only: bool = False,
    curriculum_dir: Path = None,
    high_quality_dir: Path = None
) -> list:
    """
    Load text files from synthetic, curriculum, high_quality, and raw_text repositories.
    Guarantees parsing of Claude, ChatGPT, ShareGPT, and teacher curriculum datasets.
    """
    base_data = Path(__file__).parent.parent / "data"
    if synthetic_dir is None:
        synthetic_dir = base_data / "synthetic"
    if raw_text_dir is None:
        raw_text_dir = base_data / "raw" / "text"
    if curriculum_dir is None:
        curriculum_dir = base_data / "curriculum"
    if high_quality_dir is None:
        high_quality_dir = base_data / "high_quality"

    texts = []
    
    # 1. Load Synthetic JSONL (Claude, reasoning, congruent concepts, patterns, etc.)
    if synthetic_dir.exists():
        for jsonl_file in sorted(synthetic_dir.glob("*.jsonl")):
            if not jsonl_file.is_file():
                continue
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            text = extract_text_from_record(record)
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            continue

    # 2. Load Curriculum JSONL (Council Teacher distillation, e.g. council_teacher.jsonl)
    if curriculum_dir.exists():
        for jsonl_file in sorted(curriculum_dir.glob("*.jsonl")):
            if not jsonl_file.is_file():
                continue
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            text = extract_text_from_record(record)
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            continue

    # 3. Load High-Quality JSONL (ShareGPT, Python logic, etc.)
    if high_quality_dir.exists():
        for jsonl_file in sorted(high_quality_dir.glob("*.jsonl")):
            if not jsonl_file.is_file():
                continue
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            text = extract_text_from_record(record)
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            continue

    # 4. Load Raw TXT files (SKIP if high-quality only or pure_synthetic)
    if not pure_synthetic and not high_quality_only and raw_text_dir.exists():
        for txt_file in sorted(raw_text_dir.rglob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    # Split very long books into manageable chunk paragraphs so they don't exceed max context
                    paragraphs = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 10]
                    for p in paragraphs:
                        if len(p) > 5000:  # Book chapter without newlines
                            for i in range(0, len(p), 2000):
                                texts.append(p[i:i+2000])
                        else:
                            texts.append(p)
    
    return texts


# ═══════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════
def train(args):
    print("═" * 60)
    print("  PCEI Training Loop — Physically Constrained Emergent Intelligence")
    print("═" * 60)
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[Device] {device}")
    
    from src.config import PCEIConfig
    config = PCEIConfig.load()
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path if "ckpt_path" in locals() else checkpoint_path,
        config.layers_text, config.layers_img, config.layers_audio,
        config.d_model_text, config.d_model_img, config.d_model_audio
    )
    print(f"[Topology] Detected: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")
    config.layers_text = layers_text
    config.layers_img = layers_img
    config.layers_audio = layers_audio
    config.d_model_text = d_text
    config.d_model_img = d_img
    config.d_model_audio = d_audio

    max_seq_len = getattr(args, 'max_seq_len', MAX_SEQ_LEN)
    # Init model
    nexus = PCEINexus(
        config=config,
        device=device
    )
    
    total_params = sum(p.numel() for p in nexus.parameters())
    trainable_params = sum(p.numel() for p in nexus.parameters() if p.requires_grad)
    print(f"[Model] {total_params:,} total params, {trainable_params:,} trainable")
    print(f"[Context] {max_seq_len} tokens")
    print(f"[Data] Continuous Sequence Packing: {'ENABLED' if getattr(args, 'pack_sequences', False) else 'DISABLED'}")
    print(f"[Physics] Min-SNR Loss Weighting: {'ENABLED' if getattr(args, 'min_snr', False) else 'DISABLED'}")
    
    # Optimizer — higher LR for meaningful training
    optimizer = PCEINexus.build_physics_optimizer(nexus, base_lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens
    
    # LR Scheduling: warmup then cosine decay OR WSD (Warmup-Stable-Decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR, SequentialLR
    steps_per_phase = getattr(args, 'steps_per_phase', None)
    if steps_per_phase is not None and steps_per_phase > 0:
        total_steps = steps_per_phase * 3
    else:
        total_steps = args.bootstrap_steps + args.integration_steps + args.open_steps
    
    if getattr(args, 'schedule', 'cosine') == 'wsd':
        warmup_steps = max(1, int(round(total_steps * 0.05)))
        stable_steps = max(1, int(round(total_steps * 0.80)))
        decay_steps = max(1, total_steps - warmup_steps - stable_steps)
        
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        stable_sched = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
        decay_sched = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, stable_sched, decay_sched],
            milestones=[warmup_steps, warmup_steps + stable_steps]
        )
        print(f"[Scheduler] WSD: {warmup_steps} warmup steps (5%), {stable_steps} stable max LR steps (80%), {decay_steps} cosine decay steps (15%)")
    else:
        warmup_steps = min(100, max(args.bootstrap_steps // 5, 10))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        print(f"[Scheduler] Warmup {warmup_steps} steps, cosine decay over {total_steps - warmup_steps} steps")
    
    # ═══════════════════════════════════════
    # CHECKPOINT RESUME (Automatic)
    # ═══════════════════════════════════════
    import os
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    start_step = 0
    if ckpt_path.exists():
        print(f"[Engine] FOUND. Resuming from: {ckpt_path.name} ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        # Resuming with new architecture means we allow missing keys (physics units)
        missing, unexpected = nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
        nexus.to(nexus.system_dtype)  # Re-enforce dtype — load_state_dict can leave stray float32 submodules
        if missing: print(f"  [Resume] New parameters initialized from scratch: {len(missing)}")
        
        # --- OPTIMAL WIRING: High-Quality Metabolic Flush ---
        if args.high_quality_only:
            print(f"\n[bold red]Optimal Wiring: High-Quality Mode Active.[/bold red]")
            print(f"Decaying confidence to increase plasticity for logic ingestion...")
            with torch.no_grad():
                for name, module in nexus.named_modules():
                    if hasattr(module, 'slow_field_confidence'):
                        module.slow_field_confidence.data *= 0.6 # 40% decay
                        
        if args.metabolic_flush:
            print("[System] Metabolic Flush: Resetting slow-field confidence to zero...")
            with torch.no_grad():
                for m in nexus.core.modules():
                    if isinstance(m, PlasticLayer):
                        m.slow_field_confidence.zero_()
        
        # --- ATOMIC NAN SCRUB (The Foundation Shield) ---
        with torch.no_grad():
            nan_count = 0
            for name, p in nexus.named_parameters():
                mask = torch.isnan(p) | torch.isinf(p)
                if mask.any():
                    nan_count += mask.sum().item()
                    # Apply specific floors for critical scalars
                    if "dream_strength" in name:
                        p.data.fill_(0.01)
                    else:
                        # Robust initialization for revived/scrubbed model layers
                        temp = torch.empty_like(p)
                        if p.dim() >= 2:
                            nn.init.normal_(temp, mean=0.0, std=0.02)
                        else:
                            temp.fill_(0.0)
                        p.data[mask] = temp[mask]
            
            # Also scrub buffers (where slow_field_confidence lives)
            for name, b in nexus.named_buffers():
                mask = torch.isnan(b) | torch.isinf(b)
                if mask.any():
                    nan_count += mask.sum().item()
                    b.data[mask] = 0.0
                    
            if nan_count > 0:
                print(f"  [Sanitizer] Atomic Scrub removed {nan_count} NaNs/Infs from foundation.")
        
        if 'optimizer_state_dict' in checkpoint:
            if nan_count > 0:
                print("  [Sanitizer] Corruption detected. Discarding poisoned optimizer state to prevent recurrence.")
            else:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except ValueError:
                    print("  [Warning] Optimizer state mismatch (Arch change). Re-initializing optimizer.")
        
        if 'scheduler_state_dict' in checkpoint and nan_count == 0:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  [Resume] Restored learning rate scheduler state.")
            except Exception as e:
                print(f"  [Warning] Could not restore scheduler state: {e}")

        # Restore ambient field states from checkpoint
        if 'ambient_slow_state' in checkpoint and checkpoint['ambient_slow_state'] is not None:
            nexus.ambient_field.slow_state = checkpoint['ambient_slow_state'].to(device)
            print("  [Resume] Restored ambient_field.slow_state from checkpoint.")
        if 'ambient_fast_state' in checkpoint and checkpoint['ambient_fast_state'] is not None:
            nexus.ambient_field.fast_state = checkpoint['ambient_fast_state'].to(device)
            print("  [Resume] Restored ambient_field.fast_state from checkpoint.")
        
        start_step = checkpoint.get('global_step', 0)
        print(f"  -> Restored weights. Starting at effective global step {start_step}.")
    else:
        print(f"[Engine] No checkpoint found. Initializing new model.")
            
    # BIG BANG: If all layers are at (0,0), they are trapped in a symmetry lock.
    # Inject one-time "Big Bang" jitter after resuming to enable movement.
    found_displacement = False
    for module in nexus.modules():
        if isinstance(module, PlasticLayer):
            pos = module.get_position()
            if abs(pos[0]) > 1e-3 or abs(pos[1]) > 1e-3:
                found_displacement = True
                break
    
    if not found_displacement:
        print("[Engine] Symmetry Lock detected (all layers at 0,0). Triggering Big Bang Jitter...")
        for module in nexus.modules():
            if isinstance(module, PlasticLayer):
                jitter_x = (random.random() - 0.5) * 0.1
                jitter_y = (random.random() - 0.5) * 0.1
                module.update_position(jitter_x, jitter_y)
    
    print(f"[Engine] Plotting system active: {HAS_PLOTS}")
    metrics = PCEIMetrics()
    metrics.global_step = start_step
    max_seq_len = getattr(args, 'max_seq_len', MAX_SEQ_LEN)
    config = vars(args)
    config['device'] = device
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    config['max_seq_len'] = max_seq_len
    with open(metrics.log_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Physics engine (Now integrated in PCEIModel core)
    # We no longer need manual detectors/broker in the training loop.
    
    # ═══════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════
    synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
    raw_text_dir = Path(__file__).parent.parent / "data" / "raw" / "text"
    curriculum_dir = Path(__file__).parent.parent / "data" / "curriculum"
    high_quality_dir = Path(__file__).parent.parent / "data" / "high_quality"
    
    all_texts = load_text_data(
        synthetic_dir=synthetic_dir,
        raw_text_dir=raw_text_dir,
        pure_synthetic=args.pure_synthetic,
        high_quality_only=args.high_quality_only,
        curriculum_dir=curriculum_dir,
        high_quality_dir=high_quality_dir
    )
    tokenizer = PCEITokenizer(vocab_size=args.vocab_size)
    
    # Load real multimodal data (COCO/Clotho)
    real_data_path = Path(__file__).parent.parent / "data" / "processed" / "training_database.jsonl"
    real_dataset = None
    if real_data_path.exists():
        print(f"[Schema Check] Validating {real_data_path.name}...")
        try:
            with open(real_data_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        record = json.loads(line)
                        missing = [k for k in ('id', 'type', 'text', 'image_path', 'audio_path') if k not in record]
                        if missing:
                            raise ValueError(f"Line {i+1} missing keys: {missing}")
            print("[Schema Check] PASSED.")
            
            real_dataset = PCEIDataset(real_data_path, tokenizer=tokenizer)
            print(f"[Real Data] Multi-modal pipeline active: {len(real_dataset.records)} records found.")
        except json.JSONDecodeError as e:
            print(f"[Schema Error] Corrupted JSON at line {i+1}: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"[Schema Error] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[Real Data] Error initializing dataset: {e}")
            sys.exit(1)
    
    # Text split
    random.shuffle(all_texts)
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    eval_texts = all_texts[split_idx:]
    
    # Continuous Sequence Packers (when packing is enabled)
    pack_seqs = getattr(args, 'pack_sequences', False)
    train_packer = ContinuousSequencePacker(train_texts, tokenizer, max_seq_len=max_seq_len) if pack_seqs else None
    eval_packer = ContinuousSequencePacker(eval_texts, tokenizer, max_seq_len=max_seq_len) if (pack_seqs and eval_texts) else None

    def get_text_only_batch(texts, batch_size):
        """Samples from text folders. When pack_sequences is enabled, packs documents continuously without padding zeros."""
        if pack_seqs:
            if texts is train_texts and train_packer is not None:
                return train_packer.get_batch(batch_size, device=device)
            elif texts is eval_texts and eval_packer is not None:
                return eval_packer.get_batch(batch_size, device=device)
            else:
                ad_hoc_packer = ContinuousSequencePacker(texts, tokenizer, max_seq_len=max_seq_len)
                return ad_hoc_packer.get_batch(batch_size, device=device)

        batch_tokens = []
        for _ in range(batch_size):
            text = random.choice(texts) if texts else "blank placeholder"
            # Hard Safety: Truncate to 5000 chars to prevent tokenizer hang on massive files
            safe_text = text[:5000] if isinstance(text, str) else "blank placeholder"
            batch_tokens.append(tokenizer.encode(safe_text, max_len=max_seq_len))
        
        return {
            'text': torch.stack(batch_tokens).to(device),
            'image': None,
            'audio': None
        }

    def get_multimodal_paired_batch(batch_size):
        """Samples STRICTLY from the synchronized multimodal records (Integration)."""
        if not real_dataset:
            return get_text_only_batch(train_texts, batch_size)
            
        m_batch = real_dataset.get_multimodal_batch(batch_size)
        for k in m_batch:
            if m_batch[k] is not None:
                m_batch[k] = m_batch[k].to(device)
        return m_batch

    def get_train_batch(batch_size): return get_text_only_batch(train_texts, batch_size)
    def get_eval_batch(batch_size): return get_text_only_batch(eval_texts, batch_size)
    
    # Live Dialogue Ingestion
    chat_log_dir = Path(__file__).parent.parent / "data" / "raw" / "chat_logs"
    chat_log_dir.mkdir(parents=True, exist_ok=True)
    known_log_lines = set()

    def refresh_chat_logs():
        """Peeks at chat logs and injects them into the training pool."""
        new_count = 0
        for log_file in chat_log_dir.glob("*.jsonl"):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if line not in known_log_lines:
                            data = json.loads(line)
                            # Format for training: User prompt + Model response
                            entry = f"User: {data['user']}\nModel: {data['model']}"
                            train_texts.append(entry)
                            known_log_lines.add(line)
                            new_count += 1
            except: pass
        if new_count > 0:
            print(f"  [Learning] Ingested {new_count} new lines from your conversation.")
    
    # ═══════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════
    def get_mean_confidence() -> float:
        conf_vals = []
        for _, module in nexus.core.named_modules():
            if isinstance(module, PlasticLayer):
                val = module.slow_field_confidence.item()
                # Shield: If any layer is poisoned, treat it as zero confidence
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                conf_vals.append(val)
        return sum(conf_vals) / max(len(conf_vals), 1)
    
    @torch.no_grad()
    def evaluate(num_batches: int = 5, is_multimodal: bool = False) -> dict:
        """Run eval pass on held-out data using autoregressive next-token prediction."""
        nexus.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        for _ in range(num_batches):
            if is_multimodal:
                batch = get_multimodal_paired_batch(args.batch_size)
            else:
                batch = get_eval_batch(args.batch_size)
            eval_tokens = batch['text'].to(device)
            if eval_tokens.size(1) > max_seq_len:
                eval_tokens = eval_tokens[:, :max_seq_len]
            
            logits, t_latent, _, _, _, _ = nexus.core(
                text_input=eval_tokens,
                img_input=batch.get('image'),
                audio_input=batch.get('audio'),
                diffusion_timesteps=1,
                override_timestep=0,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=get_mean_confidence()
            )
            
            if logits is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = eval_tokens[:, 1:].contiguous()
                
                # Chunked cross-entropy to prevent OOM
                seq_len = shift_logits.size(1)
                chunk_size = 128
                for c_start in range(0, seq_len, chunk_size):
                    c_end = min(c_start + chunk_size, seq_len)
                    chunk_logits = shift_logits[:, c_start:c_end, :]
                    chunk_labels = shift_labels[:, c_start:c_end]
                    
                    valid_mask = (chunk_labels != 0)
                    valid_count = valid_mask.sum().item()
                    if valid_count == 0:
                        continue
                    
                    chunk_loss = nn.functional.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        chunk_labels.reshape(-1),
                        ignore_index=0,
                        reduction='sum'
                    )
                    total_loss += chunk_loss.item()
                    
                    preds = chunk_logits.argmax(dim=-1)
                    total_correct += (preds[valid_mask] == chunk_labels[valid_mask]).sum().item()
                    total_tokens += valid_count
        
        nexus.train()
        
        avg_loss = total_loss / max(total_tokens, 1)
        accuracy = total_correct / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_accuracy": accuracy, "eval_tokens": total_tokens}
    
    def log_physics_tick(step_num: int, phase: str):
        """Run all periodic metrics, physics, and consolidation."""
        metrics.log_ternary_distributions(nexus.core)
        metrics.log_plasticity(nexus.core)
        # Note: log_gradient_norms is called before optimizer.zero_grad() in the training loop
        # Physics Status Log (Now handled by Core Resonance)
        bonds = nexus.core.broker.active_bonds
        
        # Apply physical forces once per training step to migrate layers based on bonds formed
        nexus.core.shifter.apply_forces(bonds)
        nexus.core.broker.tick_decay()
        
        metrics.log_bonds(bonds)
        # Spikes are now generated and applied during forward pass
        # Get dream_strength from the model's DreamSeeder
        ds_val = nexus.core.dream_seeder.dream_strength.item()
        metrics.log_dream_strength(ds_val)
        
        # Log state AFTER physics has moved the layers
        metrics.log_topology(nexus.core)
        
        nexus.ambient_field.tick_consolidation()
        
        structural_actions = nexus.neurogenesis.evaluate_topology()
        if structural_actions:
            spawned_layers = nexus.neurogenesis.apply_structural_changes(structural_actions)
            if spawned_layers:
                for layer in spawned_layers:
                    optimizer.add_param_group({'params': layer.parameters()})
                print("[Neurogenesis] Added spawned parameters to optimizer.")
            nexus.core.shifter.refresh_registry()
        
        summary = metrics.get_summary()
        print(f"  [{phase} {summary['session_step']:5d}] Loss: {summary['avg_recent_loss']:.4f} | "
              f"Bonds: {summary['total_bonds']} | "
              f"Dream: {ds_val:.4f} | "
              f"Conf: {get_mean_confidence():.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {summary['avg_step_time']:.3f}s", flush=True)
    
    # ═══════════════════════════════════════
    # Phase Boundaries Calculation
    # ═══════════════════════════════════════
    steps_per_phase = getattr(args, 'steps_per_phase', None)
    if getattr(args, 'boot_end', None) is not None:
        boot_end = args.boot_end
        integ_end = getattr(args, 'integ_end', None) or (boot_end + (steps_per_phase or 50000))
        open_end = getattr(args, 'open_end', None) or (integ_end + (steps_per_phase or 50000))
        print(f"\n{'═'*60}")
        print(f"  [Phased Plan] Absolute Phase Boundaries Active:")
        print(f"  Phase 1 (Text):       Steps {start_step} -> {boot_end}")
        print(f"  Phase 2 (Multimodal): Steps {boot_end} -> {integ_end}")
        print(f"  Phase 3 (Dreaming):   Steps {integ_end} -> {open_end}")
        print(f"{'═'*60}\n")
    elif steps_per_phase is not None and steps_per_phase > 0:
        boot_end = start_step + steps_per_phase
        integ_end = boot_end + steps_per_phase
        open_end = integ_end + steps_per_phase
        print(f"\n{'═'*60}")
        print(f"  [Phased Plan] Running 3x {steps_per_phase} Steps (Total: {steps_per_phase * 3})")
        print(f"  Phase 1 (Text):       Steps {start_step} -> {boot_end}")
        print(f"  Phase 2 (Multimodal): Steps {boot_end} -> {integ_end}")
        print(f"  Phase 3 (Dreaming):   Steps {integ_end} -> {open_end}")
        print(f"{'═'*60}\n")
    else:
        boot_end = args.bootstrap_steps
        integ_end = args.bootstrap_steps + args.integration_steps
        open_end = integ_end + args.open_steps

    def compute_scheduled_lr(current_step: int):
        """Calculates dynamic scheduled LR for WSD schedule in Phase 1."""
        if getattr(args, 'schedule', 'cosine') == 'wsd':
            d_start = getattr(args, 'decay_start', None)
            if d_start is None:
                d_start = 127500 if boot_end >= 127500 else int(boot_end * 0.8)
            min_lr = getattr(args, 'min_lr', 3e-5)
            if current_step < d_start:
                return args.lr
            elif current_step >= boot_end:
                return min_lr
            else:
                progress = (current_step - d_start) / max(1, (boot_end - d_start))
                cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr + cos_factor * (args.lr - min_lr)
        return None
    
    # ═══════════════════════════════════════
    # Phase 1: Bootstrap (Text)
    # ═══════════════════════════════════════
    if start_step < boot_end:
        print(f"\n{'─'*60}")
        print(f"  Phase 1: Bootstrap — WitnessMode Unimodal Ingestion")
        print(f"  Steps: {boot_end - start_step} (until global step {boot_end})")
        print(f"  Data: {len(train_texts)} train / {len(eval_texts)} eval examples")
        print(f"{'─'*60}")
        
        nexus.set_mode("WITNESS")
        
        for step in range(start_step, boot_end):
            try:
                metrics.step_start()
            
                # 1. Get Batch and prepare tensors
                batch = get_text_only_batch(train_texts, args.batch_size)
                tokens = batch['text'].to(device) # (B, T)
                
                # Truncate to max_seq_len to prevent memory explosion
                if tokens.size(1) > max_seq_len:
                    tokens = tokens[:, :max_seq_len]
                
                print(f"  ⚡ Step {step+1}/{boot_end} | Forward & Denoising ({tokens.size(1)} tokens)...", end="\r", flush=True)
                
                optimizer.zero_grad()
                
                # 2. Embed tokens to get clean latent target
                with torch.no_grad():
                    clean_latent = nexus.core.text_stack.integrate_input(tokens)
                
                # 3. Diffusion training: sample random timestep, add noise, denoise
                # CE auxiliary loss keeps the lm_head projection sharp
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, clean_latent, max_T=args.diffusion_steps,
                    ce_target=tokens, ce_weight=0.3,
                    use_min_snr=getattr(args, 'min_snr', False)
                )
                
                # Scaled Physics Loss (Metabolic constraint)
                dist_loss = nexus.core.shifter.calculate_metabolic_loss() if hasattr(nexus.core.shifter, 'calculate_metabolic_loss') else 0
                total_loss = total_loss + (dist_loss * 0.01)
                
                total_loss.backward()
                nexus.neurogenesis.monitor_gradients()
                
                if (step + 1) % args.grad_accum_steps == 0:
                    if step % 5 == 0:
                        metrics.log_gradient_norms(nexus.core)
                    torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                    
                    scheduled_lr = compute_scheduled_lr(step)
                    if scheduled_lr is not None:
                        for group in optimizer.param_groups:
                            group['initial_lr'] = scheduled_lr
                            group['lr'] = scheduled_lr
                    else:
                        scheduler.step()
                        
                    optimizer.step()
                    # Metabolic loop: update per-layer slow_field_confidence from activation similarity
                    nexus.ambient_field.tick(clean_latent.detach(), mse_val, detectors=getattr(nexus.core, 'detectors', None))
                    nexus.adjust_plasticity(optimizer)
                    optimizer.zero_grad()
                
                metrics.log_loss(mse_val, loss_text=mse_val)
                    
                # ═══════════════════════════════════════
                # LOGGING & PHYSICS  (coalesced — not every step)
                # ═══════════════════════════════════════

                # Chat log refresh every 50 steps (was every step — file I/O on hot path)
                if step % 50 == 0:
                    refresh_chat_logs()

                # Physics tick every 5 steps — bonds/shifter/neurogenesis are slow Python
                if step % 5 == 0:
                    log_physics_tick(step, "Boot")
                else:
                    summary = metrics.get_summary()
                    print(f"  [Boot  {step+1:5d}] Loss: {summary['avg_recent_loss']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}", end="\r", flush=True)
                
                if step % args.log_every == 0 and step > start_step:
                    metrics.save_snapshot(tag=f"boot_step{step}")
                    if HAS_PLOTS and step % args.plot_every == 0:
                        generate_all_plots(metrics.log_dir)
    
                if step % args.eval_every == 0 and eval_texts:
                    eval_result = evaluate()
                    metrics.eval_accuracy_window.append(eval_result['eval_accuracy'])
                    print(f"    📊 Eval — Loss: {eval_result['eval_loss']:.4f} | "
                          f"Accuracy: {eval_result['eval_accuracy']:.4f} | "
                          f"Tokens: {eval_result['eval_tokens']}")
                    
                    # Check grokking
                    grok = metrics.detect_grokking()
                    if grok["grokking_detected"]:
                        print(f"\n  🧠 GROKKING DETECTED at step {step}!")
                        metrics.save_snapshot(tag="GROKKING")
                    
                    # HOT SAVE: Allow chat_terminal to pick up latest weights
                    save_dict = {
                        'model_state_dict': nexus.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': step + 1,
                    }
                    PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=True)
                
                metrics.step_end()
                # MPS cache flush every 10 steps (was every step — stalls the MPS stream)
                if 'mps' in str(device) and step % 10 == 0:
                    torch.mps.empty_cache()
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Bootstrap! Saving SAFE_EXIT checkpoint...[/bold red]")
                save_dict = {
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }
                PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=False)
                print("[green]Physical state calcified. Safe to exit.[/green]")
                import sys; sys.exit(0)
            
        print(f"\n[Milestone] Bootstrap phase complete. Preserving weights...")
        save_dict = {
            'model_state_dict': nexus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': boot_end,
            'metrics_summary': metrics.get_summary(),
        }
        PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=True)
        
        start_step = boot_end

    # ═══════════════════════════════════════
    # Phase 2: Integration (Multimodal)
    # ═══════════════════════════════════════
    if start_step < integ_end:
        print(f"\n{'─'*60}")
        print(f"  Phase 2: Integration — Cross-Modal Ingestion")
        print(f"  Steps: {integ_end - start_step} (until global step {integ_end})")
        print(f"{'─'*60}")
        
        nexus.set_mode("WITNESS")
        real_batch_size = max(1, args.batch_size // 2)
        
        # Stabilize Integration: Lower LR
        for group in optimizer.param_groups:
            group['lr'] = 5e-5
            
        for step in range(start_step, integ_end):
            try:
                metrics.step_start()
                
                if real_dataset is not None:
                    if step % 4 == 0:
                        batch = get_text_only_batch(train_texts, args.batch_size)
                        text, img, audio = batch['text'], None, None
                    else:
                        batch = get_multimodal_paired_batch(real_batch_size)
                        text = batch['text'].to(device) if batch.get('text') is not None else None
                        if text is not None and text.size(1) > max_seq_len:
                            text = text[:, :max_seq_len]
                        img = batch['image'].to(device) if batch.get('image') is not None else None
                        audio = batch['audio'].to(device) if batch.get('audio') is not None else None
                else:
                    batch = get_text_only_batch(train_texts, args.batch_size)
                    text = batch['text'].to(device)
                    if text.size(1) > max_seq_len:
                        text = text[:, :max_seq_len]
                    img = torch.randn(args.batch_size, 196, 768, device=device, dtype=nexus.system_dtype)
                    audio = torch.randn(args.batch_size, 100, 80, device=device, dtype=nexus.system_dtype)
                
                print(f"  ⚡ Step {step+1}/{integ_end} | Integration forward & backward...", end="\r", flush=True)
                optimizer.zero_grad()
                
                # Embed text to get clean latent target
                if text is not None:
                    with torch.no_grad():
                        clean_text_latent = nexus.core.text_stack.integrate_input(text)
                else:
                    clean_text_latent = None
                
                # Diffusion training on text modality with cross-modal context
                if clean_text_latent is not None:
                    total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                        nexus, clean_text_latent, max_T=12,
                        ce_target=text, ce_weight=0.3,
                        img_input=img, audio_input=audio,
                        use_min_snr=getattr(args, 'min_snr', False)
                    )
                    
                    # Metabolic physics loss
                    metabolic_loss = torch.tensor(0.0, device=device)
                    if nexus.core.broker.active_bonds:
                        for (id_a, id_b), strength in nexus.core.broker.active_bonds.items():
                            proximity = nexus.core.shifter.calculate_proximity_reward(id_a, id_b)
                            metabolic_loss += strength * (1.0 - proximity)
                    
                    total_loss = total_loss + (metabolic_loss * 0.01)
                    total_loss.backward()
                    nexus.neurogenesis.monitor_gradients()
                    
                    if (step + 1) % args.grad_accum_steps == 0:
                        if step % 5 == 0:
                            metrics.log_gradient_norms(nexus.core)
                        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                        optimizer.step()
                        # Metabolic loop: update per-layer slow_field_confidence from activation similarity
                        nexus.ambient_field.tick(clean_text_latent.detach(), mse_val, detectors=getattr(nexus.core, 'detectors', None))
                        nexus.adjust_plasticity(optimizer)
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    metrics.log_loss(mse_val, loss_text=mse_val)
                
                # Log every step
                log_physics_tick(step, "Integ")
                
                if step % args.log_every == 0 and step > start_step:
                    metrics.save_snapshot(tag=f"integ_step{step}")
                    if HAS_PLOTS and step % args.plot_every == 0:
                        generate_all_plots(metrics.log_dir)
                    
                # HOT SAVE every 50 steps (1.82GB write — don't do this every 10)
                if step % 50 == 0 and step > 0:
                    save_dict = {
                        'model_state_dict': nexus.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': step + 1,
                    }
                    PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=True)
                
                if step % args.eval_every == 0:
                    eval_result = evaluate(is_multimodal=True)
                    metrics.eval_accuracy_window.append(eval_result['eval_accuracy'])
                    print(f"    📊 Eval (Integ) — Loss: {eval_result['eval_loss']:.4f} | "
                          f"Accuracy: {eval_result['eval_accuracy']:.4f} | "
                          f"Tokens: {eval_result['eval_tokens']}")
                    
                    # Check grokking
                    grok = metrics.detect_grokking()
                    if grok["grokking_detected"]:
                        print(f"\n  🧠 GROKKING DETECTED in Phase 2 at step {step}!")
                        metrics.save_snapshot(tag="GROKKING_P2")
                
                metrics.step_end()
                if 'mps' in str(device):
                    torch.mps.empty_cache()
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Integration! Saving SAFE_EXIT checkpoint...[/bold red]")
                save_dict = {
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }
                PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=False)
                print("[green]Physical state calcified. Safe to exit.[/green]")
                import sys; sys.exit(0)
            
        print(f"\n[Milestone] Integration phase complete. Preserving weights...")
        save_dict = {
            'model_state_dict': nexus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': integ_end,
            'metrics_summary': metrics.get_summary(),
        }
        PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=True)
        
        start_step = integ_end

    # ═══════════════════════════════════════
    # Phase 3: Open Learning (Internal Dialogue / Dreaming)
    # ═══════════════════════════════════════
    if start_step < open_end:
        print(f"\n{'─'*60}")
        print(f"  Phase 3: Internal Dialogue — Self-Resonance & Dreaming")
        print(f"  Steps: {open_end - start_step} (until global step {open_end})")
        print(f"  Data: NONE (Resonating on AmbientField)")
        print(f"{'─'*60}")
        
        nexus.set_mode("ACTIVE")
        
        for step in range(start_step, open_end):
            try:
                metrics.step_start()
                
                # Phase 3: Self-consolidation via diffusion denoising
                # Use slow_state as the clean target — the model learns to
                # denoise toward its own consolidated memory.
                optimizer.zero_grad()
                
                if nexus.ambient_field.slow_state is not None:
                    target_state = nexus.ambient_field.slow_state.detach().to(device)
                else:
                    target_state = nexus.ambient_field.get_dream_data().detach().to(device)
                
                # Collapse any batch or leading dimensions to strictly (d_text,)
                while target_state.dim() > 1:
                    target_state = target_state.mean(dim=0)
                if target_state.size(-1) != nexus.core.text_stack.d_model:
                    target_state = target_state[:nexus.core.text_stack.d_model]
                sys_dtype = next(nexus.parameters()).dtype
                target_state = target_state.to(dtype=sys_dtype)
                # Expand slow_state (1, 1, d_text) to (1, 64, d_text) as dream target
                dream_target = target_state.view(1, 1, -1).expand(1, 64, -1).clone()
                
                # Prevent Temporal Flattening: Inject temporal variance
                temporal_noise = torch.randn_like(dream_target) * (dream_target.std() * 0.2 + 1e-4)
                dream_target = dream_target + temporal_noise
                
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, dream_target, max_T=12,
                    use_min_snr=getattr(args, 'min_snr', False)
                )
                total_loss.backward()
                nexus.neurogenesis.monitor_gradients()
                
                if (step + 1) % args.grad_accum_steps == 0:
                    if step % 5 == 0:
                        metrics.log_gradient_norms(nexus.core)
                    torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    # Metabolic loop: dreaming toward slow_state — confidence rises as model
                    # successfully consolidates, progressively reducing plasticity (pruning)
                    nexus.ambient_field.tick(dream_target.detach(), mse_val, detectors=getattr(nexus.core, 'detectors', None))
                    nexus.adjust_plasticity(optimizer)
                    scheduler.step()
                    optimizer.zero_grad()
                
                metrics.log_loss(mse_val, loss_text=mse_val)
                metrics.train_loss_window.append(mse_val)
                
                # Log every step
                log_physics_tick(step, "Dream")
                    
                if step % args.eval_every == 0 and eval_texts:
                    # In dreaming phase, we monitor if semantic coherence is IMPROVING on unseen text
                    eval_result = evaluate()
                    metrics.eval_accuracy_window.append(eval_result['eval_accuracy'])
                    print(f"    📊 Dream Eval — Accuracy on Unseen Text: {eval_result['eval_accuracy']:.4f}")
    
                if step > 0 and step % args.plot_every == 0:
                    metrics.save_snapshot(tag=f"dream_step{step}")
                    if HAS_PLOTS:
                        generate_all_plots(metrics.log_dir)
                
                metrics.step_end()
                if 'mps' in str(device):
                    torch.mps.empty_cache()
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Dreaming! Saving SAFE_EXIT checkpoint...[/bold red]")
                save_dict = {
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }
                PCEINexus.save_safe_checkpoint(nexus, save_dict, Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name, create_backup=False)
                print("[green]Physical state calcified. Safe to exit.[/green]")
                import sys; sys.exit(0)
    
    # ═══════════════════════════════════════
    # FINAL SAVE
    # ═══════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Training Complete!")
    print(f"{'═'*60}")
    
    # Final eval
    if eval_texts:
        final_eval = evaluate(num_batches=10)
        print(f"  Final Eval — Loss: {final_eval['eval_loss']:.4f} | "
              f"Accuracy: {final_eval['eval_accuracy']:.4f}")
    
    metrics.save_snapshot(tag="final")
    if HAS_PLOTS:
        generate_all_plots(metrics.log_dir)
    
    # Save checkpoint
    ckpt_path = Path(__file__).parent.parent / "checkpoints"
    ckpt_path.mkdir(exist_ok=True)
    save_dict = {
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics_summary': metrics.get_summary(),
        'global_step': metrics.global_step,
        'config': config,
    }
    PCEINexus.save_safe_checkpoint(nexus, save_dict, ckpt_path / args.checkpoint_name, create_backup=True)
    
    final_summary = metrics.get_summary()
    final_summary['data_sources'] = {
        'total_texts': len(all_texts),
        'train_texts': len(train_texts),
        'eval_texts': len(eval_texts),
    }
    if eval_texts:
        final_summary['final_eval'] = final_eval
    final_summary['grokking'] = metrics.detect_grokking()
    
    with open(metrics.log_dir / "training_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Metrics:    {metrics.log_dir}/")
    print(f"  Plots:      {metrics.log_dir}/*.png")
    print(f"  Summary:    {metrics.log_dir / 'training_summary.json'}")
    
    return metrics


if __name__ == "__main__":
    import subprocess, os, platform
    _caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    parser = argparse.ArgumentParser(description="PCEI Training Loop")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--bootstrap-steps", type=int, default=500)
    parser.add_argument("--integration-steps", type=int, default=250)
    parser.add_argument("--open-steps", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--plot-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from pcei_checkpoint.pt if it exists")
    parser.add_argument("--pure-synthetic", action="store_true", default=False, help="Only train on synthetic data files")
    parser.add_argument("--high-quality-only", action="store_true", default=False, help="Only train on ShareGPT and Python logic datasets")
    
    # Scaling Arguments (Micro-Nexus Scale)
    parser.add_argument("--d-model-text", type=int, default=1536)
    parser.add_argument("--d-model-img", type=int, default=512)
    parser.add_argument("--d-model-audio", type=int, default=256)
    parser.add_argument("--layers-text", type=int, default=24)
    parser.add_argument("--layers-img", type=int, default=4)
    parser.add_argument("--layers-audio", type=int, default=4)
    parser.add_argument("--checkpoint-name", type=str, default="pcei_100m.pt", help="Which checkpoint file to load and save")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * this). Use 4+ for CUDA; 1 is fastest on MPS.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--metabolic-flush", action="store_true", default=False, help="Reset slow-field confidence to zero before training")
    
    # Training Dynamics & Scheduling
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "wsd"],
                        help="Learning rate schedule: 'cosine' (warmup + cosine) or 'wsd' (5%% warmup, 80%% stable max LR, 15%% cosine decay)")
    parser.add_argument("--pack-sequences", "--continuous-packing", dest="pack_sequences", action="store_true", default=False,
                        help="Enable continuous sequence packing: concatenate documents separated by <|endoftext|> without padding zeros")
    parser.add_argument("--min-snr", action="store_true", default=False,
                        help="Enable Min-SNR loss weighting for diffusion training steps")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum sequence length (context window)")
    parser.add_argument("--steps-per-phase", type=int, default=None,
                        help="Run each of the 3 phases (Text, Multimodal, Dreaming) for exactly this many relative steps")
    parser.add_argument("--decay-start", type=int, default=None,
                        help="Global step at which WSD cosine decay starts (e.g. 127500)")
    parser.add_argument("--boot-end", type=int, default=None,
                        help="Absolute global step for Phase 1 end (e.g. 160001)")
    parser.add_argument("--integ-end", type=int, default=None,
                        help="Absolute global step for Phase 2 end (e.g. 210001)")
    parser.add_argument("--open-end", type=int, default=None,
                        help="Absolute global step for Phase 3 end (e.g. 260001)")
    parser.add_argument("--min-lr", type=float, default=3e-5,
                        help="Minimum learning rate at the end of cosine decay")
    
    args = parser.parse_args()
    train(args)

