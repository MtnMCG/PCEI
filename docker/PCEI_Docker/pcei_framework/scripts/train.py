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


def load_text_data(synthetic_dir: Path, raw_text_dir: Path, pure_synthetic: bool = False, high_quality_only: bool = False) -> list:
    """Load text files. If high_quality_only is True, skip generic text/books."""
    texts = []
    
    # 1. Load Synthetic JSONL (Include in high-quality mode as it is reasoning-based)
    if synthetic_dir.exists():
        for jsonl_file in sorted(synthetic_dir.glob("*.jsonl")):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            text = record.get('text', '')
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            pass
                            
    # 2. Load Raw TXT files (SKIP if high-quality only or pure_synthetic)
    if not pure_synthetic and not high_quality_only and raw_text_dir.exists():
        for txt_file in sorted(raw_text_dir.rglob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    # Split very long books into manageable chunk paragraphs so they don't exceed max context
                    # If double newlines are missing, split by characters to prevent single 10MB strings
                    paragraphs = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 10]
                    for p in paragraphs:
                        if len(p) > 5000: # Book chapter without newlines
                            # Sub-split into 2000 char chunks
                            for i in range(0, len(p), 2000):
                                texts.append(p[i:i+2000])
                        else:
                            texts.append(p)
                    
    # 3. Load High-Quality (ShareGPT/Python)
    hq_dir = Path(__file__).parent.parent / "data" / "high_quality"
    if hq_dir.exists():
        for jsonl_file in sorted(hq_dir.glob("*.jsonl")):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            # Handle different formats (ShareGPT/CodeFeedback)
                            text = record.get('text', record.get('instruction', '') + "\n" + record.get('output', ''))
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            continue
    
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
    
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name
    layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, args.layers_text, args.layers_img, args.layers_audio
    )
    print(f"[Topology] Detected: {layers_text} Text | {layers_img} Img | {layers_audio} Audio | {d_text} D_Text")

    # Init model (seq len now 128)
    nexus = PCEINexus(
        vocab_size=args.vocab_size, 
        device=device,
        d_model_text=d_text,
        d_model_img=d_img,
        d_model_audio=d_audio,
        layers_text=layers_text,
        layers_img=layers_img,
        layers_audio=layers_audio
    )
    
    total_params = sum(p.numel() for p in nexus.parameters())
    trainable_params = sum(p.numel() for p in nexus.parameters() if p.requires_grad)
    print(f"[Model] {total_params:,} total params, {trainable_params:,} trainable")
    print(f"[Context] {MAX_SEQ_LEN} tokens")
    
    # Optimizer — higher LR for meaningful training
    optimizer = torch.optim.AdamW(nexus.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens
    
    # LR Scheduling: warmup then cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    total_steps = args.bootstrap_steps + args.integration_steps + args.open_steps
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
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Resuming with new architecture means we allow missing keys (physics units)
        missing, unexpected = nexus.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing: print(f"  [Resume] New parameters initialized from scratch: {len(missing)}")
        
        # --- OPTIMAL WIRING: High-Quality Metabolic Flush ---
        if args.high_quality_only:
            print(f"\n[bold red]Optimal Wiring: High-Quality Mode Active.[/bold red]")
            print(f"Decaying confidence to increase plasticity for logic ingestion...")
            with torch.no_grad():
                for name, module in nexus.named_modules():
                    if hasattr(module, 'slow_field_confidence'):
                        module.slow_field_confidence.data *= 0.6 # 40% decay
        
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
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError:
                print("  [Warning] Optimizer state mismatch (Arch change). Re-initializing optimizer.")
        
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
    config = vars(args)
    config['device'] = device
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    config['max_seq_len'] = MAX_SEQ_LEN
    with open(metrics.log_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Physics engine (Now integrated in PCEIModel core)
    # We no longer need manual detectors/broker in the training loop.
    
    # ═══════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════
    synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
    raw_text_dir = Path(__file__).parent.parent / "data" / "raw" / "text"
    
    all_texts = load_text_data(synthetic_dir, raw_text_dir, pure_synthetic=args.pure_synthetic, high_quality_only=args.high_quality_only)
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
    
    def get_text_only_batch(texts, batch_size):
        """Samples STRICTLY from the synthetic/user text folders (Bootstrapping)."""
        batch_tokens = []
        for _ in range(batch_size):
            text = random.choice(texts) if texts else "blank placeholder"
            # Hard Safety: Truncate to 5000 chars to prevent tokenizer hang on massive files
            safe_text = text[:5000] if isinstance(text, str) else "blank placeholder"
            batch_tokens.append(tokenizer.encode(safe_text))
        
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
        """Run eval pass on held-out data using MASKED DENOISING (matches training objective)."""
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
            target = eval_tokens.clone()
            
            # Apply same masking as training (30% fixed for eval consistency)
            mask_ratio = 0.30
            mask = torch.rand(eval_tokens.shape, device=device) < mask_ratio
            mask[:, 0] = False  # Never mask first token
            
            input_tokens = eval_tokens.clone()
            input_tokens[mask] = tokenizer.pad_id
            
            logits, t_latent, _, _, _, _ = nexus.core(
                text_input=input_tokens,
                img_input=batch.get('image'),
                audio_input=batch.get('audio'),
                diffusion_timesteps=1,
                slow_state=nexus.ambient_field.slow_state,
                mean_confidence=get_mean_confidence()
            )
            
            if logits is not None:
                # Loss only on masked positions (same as training)
                loss_all = nn.functional.cross_entropy(
                    logits.transpose(1, 2),
                    target,
                    reduction='none'
                )
                masked_loss = (loss_all * mask.float()).sum() / (mask.sum() + 1e-6)
                total_loss += masked_loss.item()
                
                # Accuracy: on masked positions only
                preds = logits.argmax(dim=-1)
                if mask.sum() > 0:
                    total_correct += (preds[mask] == target[mask]).sum().item()
                    total_tokens += mask.sum().item()
        
        nexus.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_accuracy": accuracy, "eval_tokens": total_tokens}
    
    def log_physics_tick(step_num: int, phase: str):
        """Run all periodic metrics, physics, and consolidation."""
        metrics.log_ternary_distributions(nexus.core)
        metrics.log_plasticity(nexus.core)
        metrics.log_gradient_norms(nexus.core)
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
            nexus.neurogenesis.apply_structural_changes(structural_actions)
            nexus.core.shifter.refresh_registry()
        
        summary = metrics.get_summary()
        print(f"  [{phase} {summary['session_step']:5d}] Loss: {summary['avg_recent_loss']:.4f} | "
              f"Bonds: {summary['total_bonds']} | "
              f"Dream: {ds_val:.4f} | "
              f"Conf: {get_mean_confidence():.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {summary['avg_step_time']:.3f}s")
    
    # ═══════════════════════════════════════
    # Phase 1: Bootstrap
    # ═══════════════════════════════════════
    if start_step < args.bootstrap_steps:
        print(f"\n{'─'*60}")
        print(f"  Phase 1: Bootstrap — WitnessMode Unimodal Ingestion")
        print(f"  Steps: {args.bootstrap_steps}")
        print(f"  Data: {len(train_texts)} train / {len(eval_texts)} eval examples")
        print(f"{'─'*60}")
        
        nexus.set_mode("WITNESS")
        
        for step in range(start_step, args.bootstrap_steps):
            try:
                metrics.step_start()
            
                # 1. Get Batch and prepare tensors
                batch = get_text_only_batch(train_texts, args.batch_size)
                tokens = batch['text'].to(device) # (B, T)
                
                optimizer.zero_grad()
                
                # 2. Embed tokens to get clean latent target
                with torch.no_grad():
                    clean_latent = nexus.core.text_stack.integrate_input(tokens)
                
                # 3. Diffusion training: sample random timestep, add noise, denoise
                # CE auxiliary loss keeps the lm_head projection sharp
                total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                    nexus, clean_latent, max_T=12,
                    ce_target=tokens, ce_weight=0.3
                )
                
                # Scaled Physics Loss (Metabolic constraint)
                dist_loss = nexus.core.shifter.calculate_metabolic_loss() if hasattr(nexus.core.shifter, 'calculate_metabolic_loss') else 0
                total_loss = total_loss + (dist_loss * 0.01)
                
                total_loss.backward()
                
                if (step + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                metrics.log_loss(mse_val, loss_text=mse_val)
                    
                # ═══════════════════════════════════════
                # LOGGING & PHYSICS
                # ═══════════════════════════════════════
                refresh_chat_logs() # Learn from chat (fast)
                log_physics_tick(step, "Boot")
                
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
                    torch.save({
                        'model_state_dict': nexus.state_dict(),
                        'global_step': metrics.global_step,
                    }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
                    os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
                
                metrics.step_end()
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Bootstrap! Saving SAFE_EXIT checkpoint...[/bold red]")
                torch.save({
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
                os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
                print("[green]Physical state calcified. Safe to exit.[/green]")
                import sys; sys.exit(0)
            
        print(f"\n[Milestone] Bootstrap phase complete. Preserving weights...")
        torch.save({
            'model_state_dict': nexus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': args.bootstrap_steps,
            'metrics_summary': metrics.get_summary(),
        }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
        os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
        
        start_step = args.bootstrap_steps

    # ═══════════════════════════════════════
    # Phase 2: Integration
    # ═══════════════════════════════════════
    integ_end = args.bootstrap_steps + args.integration_steps
    if real_dataset is not None and start_step < integ_end:
        print(f"\n{'─'*60}")
        print(f"  Phase 2: Integration — Cross-Modal Ingestion")
        print(f"  Steps: {args.integration_steps}")
        print(f"{'─'*60}")
        
        nexus.set_mode("WITNESS")
        real_batch_size = max(1, args.batch_size // 2)
        
        # Stabilize Integration: Lower LR
        for group in optimizer.param_groups:
            group['lr'] = 5e-5
            
        for step in range(start_step, integ_end):
            try:
                metrics.step_start()
                
                if step % 4 == 0:
                    batch = get_text_only_batch(train_texts, args.batch_size)
                    text, img, audio = batch['text'], None, None
                else:
                    batch = get_multimodal_paired_batch(real_batch_size)
                    text = batch['text'].to(device) if batch.get('text') is not None else None
                    img = batch['image'].to(device) if batch.get('image') is not None else None
                    audio = batch['audio'].to(device) if batch.get('audio') is not None else None
                
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
                        img_input=img, audio_input=audio
                    )
                    
                    # Metabolic physics loss
                    metabolic_loss = torch.tensor(0.0, device=device)
                    if nexus.core.broker.active_bonds:
                        for (id_a, id_b), strength in nexus.core.broker.active_bonds.items():
                            proximity = nexus.core.shifter.calculate_proximity_reward(id_a, id_b)
                            metabolic_loss += strength * (1.0 - proximity)
                    
                    total_loss = total_loss + (metabolic_loss * 0.01)
                    total_loss.backward()
                    
                    if (step + 1) % args.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                        optimizer.step()
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
                    torch.save({
                        'model_state_dict': nexus.state_dict(),
                        'global_step': metrics.global_step,
                    }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
                    os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
                
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
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Integration! Saving SAFE_EXIT checkpoint...[/bold red]")
                torch.save({
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
                os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
                print("[green]Physical state calcified. Safe to exit.[/green]")
                import sys; sys.exit(0)
            
        print(f"\n[Milestone] Integration phase complete. Preserving weights...")
        torch.save({
            'model_state_dict': nexus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': integ_end,
            'metrics_summary': metrics.get_summary(),
        }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
        os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
        
        start_step = integ_end

    # ═══════════════════════════════════════
    # Phase 3: Open Learning (Internal Dialogue / Dreaming)
    # ═══════════════════════════════════════
    open_end = args.bootstrap_steps + args.integration_steps + args.open_steps
    if start_step < open_end:
        print(f"\n{'─'*60}")
        print(f"  Phase 3: Internal Dialogue — Self-Resonance & Dreaming")
        print(f"  Steps: {args.open_steps}")
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
                    # Expand slow_state (d_text,) to (1, 64, d_text) as dream target
                    dream_target = target_state.view(1, 1, -1).expand(1, 64, -1)
                    
                    total_loss, mse_val, sampled_t, noise_scale = diffusion_training_step(
                        nexus, dream_target, max_T=12
                    )
                    total_loss.backward()
                    
                    if (step + 1) % args.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=args.grad_clip)
                        optimizer.step()
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
                
            except KeyboardInterrupt:
                print(f"\n[bold red]Interrupt received during Dreaming! Saving SAFE_EXIT checkpoint...[/bold red]")
                torch.save({
                    'model_state_dict': nexus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': step,
                    'metrics_summary': metrics.get_summary(),
                }, str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp')
                os.replace(str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name) + '.tmp', Path(__file__).parent.parent / "checkpoints" / args.checkpoint_name)
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
    torch.save({
        'model_state_dict': nexus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics_summary': metrics.get_summary(),
        'global_step': metrics.global_step,
        'config': config,
    }, str(ckpt_path / args.checkpoint_name) + '.tmp')
    os.replace(str(ckpt_path / args.checkpoint_name) + '.tmp', ckpt_path / args.checkpoint_name)
    
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
    parser.add_argument("--d-model-text", type=int, default=128)
    parser.add_argument("--d-model-img", type=int, default=128)
    parser.add_argument("--d-model-audio", type=int, default=64)
    parser.add_argument("--layers-text", type=int, default=2)
    parser.add_argument("--layers-img", type=int, default=1)
    parser.add_argument("--layers-audio", type=int, default=1)
    parser.add_argument("--checkpoint-name", type=str, default="pcei_micro.pt", help="Which checkpoint file to load and save")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    
    args = parser.parse_args()
    train(args)
