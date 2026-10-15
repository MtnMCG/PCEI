"""
PCEI Deep Checkpoint Probe — Forensic analysis of pcei_500m.pt
Examines weight statistics, dead layers, gradient capacity, 
embedding coherence, and generation quality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "pcei_500m.pt"

def section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def main():
    section("1. LOADING CHECKPOINT")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    
    print(f"  Keys in checkpoint: {list(ckpt.keys())}")
    print(f"  Global step saved:  {ckpt.get('global_step', 'NOT STORED')}")
    if 'config' in ckpt:
        print(f"  Config:             {ckpt['config']}")
    if 'metrics_summary' in ckpt:
        print(f"  Metrics Summary:    {ckpt['metrics_summary']}")
    print(f"  Optimizer state:    {'YES' if 'optimizer_state_dict' in ckpt else 'NO'}")
    
    total_params = sum(p.numel() for p in sd.values())
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Checkpoint size:    {CKPT_PATH.stat().st_size / 1e9:.2f} GB")

    # ─── 2. PARAMETER ANATOMY ───────────────────────────────────
    section("2. PARAMETER ANATOMY (by component)")
    
    component_stats = defaultdict(lambda: {"count": 0, "params": 0, "nan": 0, "zero_frac": 0.0, "keys": []})
    
    for key, tensor in sd.items():
        # Determine component
        parts = key.split('.')
        if 'text_stack' in key:
            comp = "TextStack"
        elif 'img_stack' in key:
            comp = "ImageStack"
        elif 'audio_stack' in key:
            comp = "AudioStack"
        elif 'boundary' in key:
            comp = "BoundaryLayer"
        elif 'dream_seeder' in key:
            comp = "DreamSeeder"
        elif 'broker' in key:
            comp = "EntanglementBroker"
        elif 'shifter' in key:
            comp = "TopologyShifter"
        elif 'ambient_field' in key:
            comp = "AmbientField"
        elif 'response_gate' in key:
            comp = "ResponseGate"
        elif 'neurogenesis' in key:
            comp = "Neurogenesis"
        else:
            comp = "Other"
        
        c = component_stats[comp]
        c["count"] += 1
        c["params"] += tensor.numel()
        c["nan"] += torch.isnan(tensor).sum().item() + torch.isinf(tensor).sum().item()
        if tensor.numel() > 0:
            c["zero_frac"] += (tensor == 0).float().mean().item()
        c["keys"].append(key)
    
    for comp, stats in sorted(component_stats.items(), key=lambda x: -x[1]["params"]):
        avg_zero = stats["zero_frac"] / max(stats["count"], 1)
        status = "🟢" if stats["nan"] == 0 and avg_zero < 0.5 else ("🟡" if stats["nan"] == 0 else "🔴")
        print(f"  {status} {comp:20s} | {stats['params']:>12,} params | {stats['count']:3d} tensors | "
              f"NaN/Inf: {stats['nan']:5d} | Avg Zero%: {avg_zero*100:.1f}%")

    # ─── 3. LAYER-BY-LAYER WEIGHT HEALTH ──────────────────────────
    section("3. TEXT STACK — LAYER-BY-LAYER WEIGHT HEALTH")
    
    text_layers = set()
    for key in sd.keys():
        if 'core.text_stack.layers.' in key:
            idx = key.split('core.text_stack.layers.')[1].split('.')[0]
            text_layers.add(idx)
    
    for layer_idx in sorted(text_layers, key=int):
        prefix = f"core.text_stack.layers.{layer_idx}"
        layer_params = {k: v for k, v in sd.items() if k.startswith(prefix)}
        
        total_numel = sum(v.numel() for v in layer_params.values())
        nan_count = sum(torch.isnan(v).sum().item() + torch.isinf(v).sum().item() for v in layer_params.values())
        zero_count = sum((v == 0).sum().item() for v in layer_params.values())
        zero_pct = zero_count / max(total_numel, 1) * 100
        
        # Weight magnitude stats
        all_weights = torch.cat([v.float().flatten() for v in layer_params.values() if 'weight' in v.__class__.__name__ or True])
        
        status = "DEAD 💀" if zero_pct > 90 else ("WEAK 🟡" if zero_pct > 50 else "ALIVE 🟢")
        print(f"  Layer {layer_idx:2s} | {total_numel:>10,} params | Zero: {zero_pct:5.1f}% | "
              f"Mean: {all_weights.mean():.6f} | Std: {all_weights.std():.6f} | "
              f"Min: {all_weights.min():.4f} | Max: {all_weights.max():.4f} | {status}")

    # ─── 4. CRITICAL SUBSYSTEM INSPECTION ────────────────────────
    section("4. CRITICAL SUBSYSTEM INSPECTION")
    
    # Mamba A_log values  
    print("\n  ── Mamba A_log (State Decay Matrix) ──")
    for key, val in sd.items():
        if 'A_log' in key and 'text_stack' in key:
            A = -torch.exp(val.float())
            print(f"    {key.split('text_stack.')[1][:40]:40s} | A range: [{A.min():.4f}, {A.max():.4f}] | mean: {A.mean():.4f}")
    
    # Mamba dt_proj bias (timestep sensitivity)
    print("\n  ── Mamba dt_proj.bias (Timestep Sensitivity) ──")
    for key, val in sd.items():
        if 'dt_proj.bias' in key and 'text_stack' in key:
            dt = F.softplus(val.float())
            print(f"    {key.split('text_stack.')[1][:40]:40s} | dt range: [{dt.min():.6f}, {dt.max():.6f}] | mean: {dt.mean():.6f}")
    
    # DreamSeeder dream_strength
    print("\n  ── DreamSeeder ──")
    for key, val in sd.items():
        if 'dream_strength' in key:
            print(f"    {key}: {val.item():.6f}")
    for key, val in sd.items():
        if 'dream_seeder' in key and 'weight' in key:
            print(f"    {key[:50]:50s} | std: {val.std():.6f} | mean: {val.mean():.6f}")
    
    # Slow field confidence
    print("\n  ── Slow Field Confidence (Plasticity) ──")
    for key, val in sd.items():
        if 'slow_field_confidence' in key:
            c = val.item()
            bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
            print(f"    {key.split('core.')[1][:40]:40s} | {c:.4f} [{bar}]")
    
    # Layer positions (topology)
    print("\n  ── Layer Positions (Topology Shifts) ──")
    positions = {}
    for key, val in sd.items():
        if 'pos_x' in key or 'pos_y' in key or 'pos_z' in key:
            layer_name = key.rsplit('.', 1)[0]
            if layer_name not in positions:
                positions[layer_name] = {}
            coord = key.rsplit('.', 1)[1]
            positions[layer_name][coord] = val.item()
    
    for layer_name, coords in sorted(positions.items()):
        x = coords.get('pos_x', 0)
        y = coords.get('pos_y', 0)
        z = coords.get('pos_z', 0)
        # Check if displaced from origin
        displaced = abs(x) > 0.01 or abs(y) > 0.01
        marker = "📍 MOVED" if displaced else "📌 stuck"
        short_name = layer_name.split('core.')[-1][:35]
        print(f"    {short_name:35s} | x={x:+.4f} y={y:+.4f} z={z:+.1f} | {marker}")

    # ─── 5. EMBEDDING COHERENCE ──────────────────────────────────
    section("5. EMBEDDING ANALYSIS")
    
    emb_key = 'core.text_stack.embedding.weight'
    if emb_key in sd:
        emb = sd[emb_key].float()
        print(f"  Shape: {emb.shape} (vocab_size x d_model)")
        print(f"  Mean:  {emb.mean():.6f}")
        print(f"  Std:   {emb.std():.6f}")
        print(f"  Norm range: [{emb.norm(dim=1).min():.4f}, {emb.norm(dim=1).max():.4f}]")
        print(f"  Zero rows: {(emb.norm(dim=1) == 0).sum().item()}")
        
        # Check if embeddings are differentiated (not just noise)
        # If all embeddings are too similar, the model can't distinguish tokens
        random_pairs = torch.randint(0, emb.shape[0], (100, 2))
        cosine_sims = F.cosine_similarity(emb[random_pairs[:, 0]], emb[random_pairs[:, 1]], dim=1)
        print(f"  Random-pair cosine similarity: mean={cosine_sims.mean():.4f} std={cosine_sims.std():.4f}")
        print(f"    → {'GOOD: Embeddings are differentiated' if cosine_sims.std() > 0.1 else 'BAD: Embeddings look uniform/noisy'}")
        
        # Check specific tokens
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        test_words = ["the", "hello", "france", "philosophy", "is", "and"]
        print(f"\n  ── Token Embedding Norms ──")
        for word in test_words:
            tid = enc.encode(word)[0]
            norm = emb[tid].norm().item()
            print(f"    '{word}' (id={tid}): norm={norm:.4f}")
        
        # Nearest neighbor check: does "france" cluster near "germany"?
        print(f"\n  ── Nearest Neighbors (cosine) ──")
        test_queries = ["france", "hello", "the"]
        for word in test_queries:
            tid = enc.encode(word)[0]
            query = emb[tid].unsqueeze(0)
            sims = F.cosine_similarity(query, emb, dim=1)
            sims[tid] = -1  # exclude self
            top5_ids = sims.topk(5).indices.tolist()
            neighbors = []
            for nid in top5_ids:
                try:
                    decoded = enc.decode([nid])
                    neighbors.append(f"'{decoded}'({sims[nid]:.3f})")
                except:
                    neighbors.append(f"[{nid}]({sims[nid]:.3f})")
            print(f"    '{word}' → {', '.join(neighbors)}")

    # ─── 6. LM HEAD ANALYSIS ────────────────────────────────────
    section("6. LM HEAD (Output Projection)")
    
    lm_head_key = 'core.text_stack.lm_head.weight'
    if lm_head_key in sd:
        lm = sd[lm_head_key].float()
        print(f"  Shape: {lm.shape}")
        print(f"  Mean:  {lm.mean():.6f}")
        print(f"  Std:   {lm.std():.6f}")
        
        # Check output bias: what tokens does the model currently favor?
        # Simulate logits from a zero input
        fake_input = torch.zeros(1, lm.shape[1])
        logits = F.linear(fake_input, lm)
        probs = F.softmax(logits[0], dim=-1)
        
        top20_ids = probs.topk(20).indices.tolist()
        print(f"\n  ── Top 20 Default Tokens (from zero input) ──")
        for i, tid in enumerate(top20_ids):
            try:
                decoded = enc.decode([tid])
                print(f"    {i+1:2d}. '{decoded}' (id={tid}, prob={probs[tid]:.6f})")
            except:
                print(f"    {i+1:2d}. [id={tid}] (prob={probs[tid]:.6f})")
        
        # Entropy of default distribution
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = math.log(probs.shape[0])
        print(f"\n  Default distribution entropy: {entropy:.2f} / {max_entropy:.2f} ({entropy/max_entropy*100:.1f}%)")
        print(f"    → {'GOOD: Peaked (model has preferences)' if entropy < max_entropy * 0.9 else 'BAD: Near-uniform (model is guessing)'}")

    # ─── 7. TERNARY WEIGHT ANALYSIS ──────────────────────────────
    section("7. TERNARY WEIGHTS (Quantization Health)")
    
    ternary_keys = [k for k in sd.keys() if 'ternary' in k and 'weight' in k]
    if ternary_keys:
        for key in ternary_keys[:6]:  # Sample first 6
            w = sd[key].float()
            # Check ternary-ness: values should cluster around {-1, 0, +1}
            near_neg1 = ((w + 1).abs() < 0.1).float().mean().item()
            near_zero = (w.abs() < 0.1).float().mean().item()
            near_pos1 = ((w - 1).abs() < 0.1).float().mean().item()
            ternary_pct = (near_neg1 + near_zero + near_pos1) * 100
            short = key.split('core.')[-1][:50]
            print(f"  {short:50s} | Ternary%: {ternary_pct:5.1f}% | "
                  f"-1: {near_neg1*100:.0f}% | 0: {near_zero*100:.0f}% | +1: {near_pos1*100:.0f}%")
    else:
        print("  No ternary weight keys found.")

    # ─── 8. GENERATION TEST ──────────────────────────────────────
    section("8. RAW LOGIT GENERATION TEST (No diffusion, pure forward)")
    
    try:
        from src.pcei import PCEINexus
        from scripts.train import PCEITokenizer
        
        layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(
            CKPT_PATH, 2, 1, 1
        )
        
        nexus = PCEINexus(
            vocab_size=50257, device="cpu",
            d_model_text=d_text, d_model_img=d_img, d_model_audio=d_audio,
            layers_text=layers_text, layers_img=layers_img, layers_audio=layers_audio
        )
        nexus.load_state_dict(sd, strict=False)
        nexus.eval()
        
        tokenizer = PCEITokenizer()
        
        prompts = [
            "The capital of France is",
            "Hello, my name is",
            "One plus one equals",
            "Philosophy is the study of",
        ]
        
        for prompt in prompts:
            ids = tokenizer.enc.encode(prompt, allowed_special={"<|endoftext|>"})
            ids = [tid if tid != 0 else 50256 for tid in ids]
            input_t = torch.tensor([ids], dtype=torch.long)
            
            with torch.no_grad():
                latent = nexus.core.text_stack.integrate_input(input_t)
                
                # Single diffusion step
                conf_vals = [m.slow_field_confidence.item() for m in nexus.core.modules() if hasattr(m, 'slow_field_confidence')]
                mean_conf = sum(conf_vals) / max(len(conf_vals), 1)
                
                _, denoised, _, _, _, _ = nexus.core(
                    text_latent=latent,
                    diffusion_timesteps=1,
                    slow_state=nexus.ambient_field.slow_state,
                    mean_confidence=mean_conf
                )
                
                logits = nexus.core.text_stack.decode_output(denoised)
                
                # Greedy decode
                pred_ids = logits[0].argmax(dim=-1).tolist()
                pred_ids = [0 if tid == 50256 else tid for tid in pred_ids]
                
                decoded = tokenizer.enc.decode(pred_ids)
                
                # Also show top-5 for last position
                last_logits = logits[0, -1]
                top5 = last_logits.topk(5)
                top5_tokens = []
                for tid_t, score in zip(top5.indices.tolist(), top5.values.tolist()):
                    try:
                        t = tokenizer.enc.decode([tid_t])
                        top5_tokens.append(f"'{t}'({score:.2f})")
                    except:
                        top5_tokens.append(f"[{tid_t}]({score:.2f})")
                
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  Greedy: \"{decoded[:80]}\"")
            print(f"  Top-5 next: {', '.join(top5_tokens)}")
            
            # Check if output is just repeating input
            input_decoded = tokenizer.enc.decode(ids)
            if decoded.strip() == input_decoded.strip():
                print(f"  ⚠️  Output is just echoing the input (identity mapping)")
            
    except Exception as e:
        print(f"  Generation test failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── 9. DIAGNOSIS SUMMARY ────────────────────────────────────
    section("9. DIAGNOSIS SUMMARY")
    
    # Collect verdicts
    issues = []
    positives = []
    
    # Check global step
    gs = ckpt.get('global_step', 0)
    if gs == 0:
        issues.append(f"Global step is 0 — the resonance chat resets step to 0 on every save")
    else:
        positives.append(f"Trained for {gs} steps")
    
    # Check NaN contamination
    total_nan = sum(stats["nan"] for stats in component_stats.values())
    if total_nan > 0:
        issues.append(f"{total_nan} NaN/Inf values found across model")
    else:
        positives.append("No NaN/Inf contamination")
    
    # Check embedding health
    if emb_key in sd:
        emb_std = sd[emb_key].float().std().item()
        if emb_std < 0.01:
            issues.append(f"Embedding std is {emb_std:.6f} — nearly collapsed")
        elif emb_std < 0.05:
            issues.append(f"Embedding std is {emb_std:.6f} — quite compressed")
        else:
            positives.append(f"Embedding std healthy: {emb_std:.4f}")
    
    # Check if optimizer state is missing
    if 'optimizer_state_dict' not in ckpt:
        issues.append("No optimizer state — momentum/Adam buffers lost on resume")
    
    # Check training data scale (from what we know)
    print("\n  🟢 POSITIVE SIGNALS:")
    for p in positives:
        print(f"    ✓ {p}")
    
    print("\n  🔴 ISSUES:")
    for i in issues:
        print(f"    ✗ {i}")
    
    print(f"\n  ── OVERALL VERDICT ──")
    print(f"  The model has {total_params:,} parameters with d_model={d_text}.")
    print(f"  At {d_text} dimensions and {len(text_layers)} text layers, this is a")
    print(f"  {'large' if total_params > 400_000_000 else 'medium'}-scale architecture.")
    print()
    
    if gs < 10000:
        needed = "100K-500K+ steps on billions of tokens"
    elif gs < 50000:
        needed = "50K-200K more steps"
    else:
        needed = "more high-quality data and longer training"
    
    print(f"  To reach coherent text generation, this model needs: {needed}")
    print(f"  Current state: The model is producing random token salad because")
    print(f"  the weights have not converged enough to learn language statistics.")

if __name__ == "__main__":
    main()
