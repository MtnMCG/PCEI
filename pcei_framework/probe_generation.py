"""
Quick probe v3: handle NaN logits gracefully, diagnose the problem.
"""
import torch
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer
import tiktoken

device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

def decode_ids(ids):
    ids = [i for i in ids if i not in (0, 50256)]
    try:
        return enc.decode(ids)
    except:
        return "<decode_error>"

print(f"Device: {device}")
print("Loading pcei_500m.pt...")
t0 = time.time()

ckpt_path = Path("checkpoints/pcei_100m.pt")
layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(ckpt_path, 2, 1, 1)
print(f"Topology: {layers_text}L text | {layers_img}L img | {layers_audio}L audio | {d_text}D")

nexus = PCEINexus(
    vocab_size=50257, device=device,
    d_model_text=d_text, d_model_img=d_img, d_model_audio=d_audio,
    layers_text=layers_text, layers_img=layers_img, layers_audio=layers_audio
)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
nexus.to(nexus.system_dtype)

with torch.no_grad():
    nan_count = 0
    for p in nexus.parameters():
        mask = torch.isnan(p) | torch.isinf(p)
        if mask.any():
            nan_count += mask.sum().item()
            p.data[mask] = 0.0
    for b in nexus.buffers():
        mask = torch.isnan(b) | torch.isinf(b)
        if mask.any():
            nan_count += mask.sum().item()
            b.data[mask] = 0.0

print(f"Loaded in {time.time()-t0:.1f}s | NaNs cleaned: {nan_count}")

# === Confidence state ===
conf_vals = []
for _, m in nexus.named_modules():
    if isinstance(m, PlasticLayer):
        conf_vals.append(m.slow_field_confidence.item())
mean_conf = sum(conf_vals) / max(len(conf_vals), 1)
print(f"\n{'='*60}")
print(f"CONFIDENCE STATE")
print(f"{'='*60}")
print(f"  Mean:  {mean_conf:.4f}")
print(f"  Min:   {min(conf_vals):.4f}")
print(f"  Max:   {max(conf_vals):.4f}")
print(f"  Layers: {len(conf_vals)}")
print(f"  Learning capacity: {(1-mean_conf)*100:.1f}%")
print(f"  Bond count: {len(nexus.core.broker.active_bonds)}")

nexus.eval()
embedding  = nexus.core.text_stack.embedding
lm_head    = nexus.core.text_stack.lm_head
norm_final = nexus.core.text_stack.norm_final

# === LM Head sanity ===
print(f"\n{'='*60}")
print(f"LM HEAD DIAGNOSTICS")
print(f"{'='*60}")
with torch.no_grad():
    # Check weights
    lmh_w = lm_head.weight
    print(f"  lm_head.weight: shape={tuple(lmh_w.shape)}")
    print(f"    nan count: {torch.isnan(lmh_w).sum().item()}")
    print(f"    inf count: {torch.isinf(lmh_w).sum().item()}")
    print(f"    mean: {lmh_w.mean().item():.6f}")
    print(f"    std:  {lmh_w.std().item():.6f}")
    print(f"    max abs: {lmh_w.abs().max().item():.4f}")

    norm_w = norm_final.weight
    print(f"  norm_final.weight mean: {norm_w.mean().item():.4f}  std: {norm_w.std().item():.6f}")
    print(f"  norm_final.bias  mean: {norm_final.bias.mean().item():.4f}")

    # Pass a zero latent through norm then lm_head
    zero_lat = torch.zeros(1, 1, d_text, device=device)
    normed = norm_final(zero_lat)
    logits0 = lm_head(normed).squeeze()
    print(f"\n  Logits from ZERO latent:")
    print(f"    nan: {torch.isnan(logits0).sum().item()}  inf: {torch.isinf(logits0).sum().item()}")
    print(f"    mean: {logits0.mean().item():.4f}  std: {logits0.std().item():.6f}  max: {logits0.max().item():.4f}")

    # Pass random latent
    rand_lat = torch.randn(1, 1, d_text, device=device)
    normed2 = norm_final(rand_lat)
    logits1 = lm_head(normed2).squeeze()
    print(f"\n  Logits from RANDN latent:")
    print(f"    nan: {torch.isnan(logits1).sum().item()}  inf: {torch.isinf(logits1).sum().item()}")
    print(f"    mean: {logits1.mean().item():.4f}  std: {logits1.std().item():.6f}  max: {logits1.max().item():.4f}")

    if not (torch.isnan(logits1).any() or torch.isinf(logits1).any()):
        probs1 = torch.softmax(logits1, dim=-1)
        entropy = -(probs1 * (probs1 + 1e-9).log()).sum().item()
        print(f"    entropy: {entropy:.2f} bits (max={torch.log(torch.tensor(50257.0)).item():.2f})")
        top5_p, top5_i = torch.topk(probs1, 5)
        print(f"    top-5: {[(enc.decode([i.item()]), f'{p.item():.4f}') for i, p in zip(top5_i, top5_p)]}")

# === Text stack layer-by-layer NaN trace ===
print(f"\n{'='*60}")
print(f"TEXT STACK LAYER-BY-LAYER TRACE (single token 'Hello')")
print(f"{'='*60}")

hello_id = enc.encode("Hello")[0]
input_ids = torch.tensor([[hello_id]], dtype=torch.long, device=device)

with torch.no_grad():
    x = embedding(input_ids).float()
    print(f"  Embedding output: nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()} mean={x.mean().item():.4f} std={x.std().item():.4f}")

    t_val = torch.tensor([0.5], device=device)

    from src.modules.adaptation import TTTAdapter
    for i, layer in enumerate(nexus.core.text_stack.layers):
        inner = layer.module if isinstance(layer, TTTAdapter) else layer
        try:
            if isinstance(layer, TTTAdapter):
                x_out = layer(x, t_val, is_training=True) # BYPASS TTT NaN CRASH
            else:
                x_out = layer(x, t_val)
            has_nan = torch.isnan(x_out).any().item()
            has_inf = torch.isinf(x_out).any().item()
            print(f"  Layer {i:2d} ({inner.layer_id}): nan={has_nan} inf={has_inf} mean={x_out.mean().item():.4f} std={x_out.std().item():.4f}")
            x = x_out
        except Exception as e:
            print(f"  Layer {i:2d}: CRASH — {e}")
            break

    # Final norm + lm_head
    normed_final = norm_final(x)
    print(f"  After norm_final: nan={torch.isnan(normed_final).any().item()} mean={normed_final.mean().item():.4f}")
    final_logits = lm_head(normed_final).squeeze()
    print(f"  Final logits: nan={torch.isnan(final_logits).any().item()} inf={torch.isinf(final_logits).any().item()} max={final_logits.max().item():.4f}")

# === Generation attempt with safety clamp ===
print(f"\n{'='*60}")
print(f"GENERATION PROBE (with NaN safety clamp)")
print(f"{'='*60}")

prompts = [
    "USER: Hello, how are you?\\nASSISTANT:"
]

DEPTH = 8
GEN_TOKENS = 10

for prompt in prompts:
    print(f"\\nPROMPT: {prompt.strip()}", flush=True)
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    ids = [i if i != 0 else 50256 for i in ids]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        emb = embedding(input_ids).float()

    generated_ids = []
    t_start = time.time()
    aborted = False

    with torch.no_grad():
        for step in range(GEN_TOKENS):
            noise = torch.randn_like(emb) * 0.05
            latent = emb + noise

            for d in range(DEPTH):
                t_val = torch.tensor([1.0 - d / max(DEPTH, 1)], device=device)
                t_batch = t_val.expand(latent.shape[0])
                latent = nexus.core.text_stack(latent, t_batch)
                latent = torch.clamp(latent, -20.0, 20.0)

            logits = lm_head(norm_final(latent[:, -1, :]))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            logits = torch.clamp(logits / 0.8, min=-50.0, max=50.0)

            top_k = 50
            top_vals, top_idxs = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)

            if torch.isnan(probs).any() or (probs < 0).any():
                print(f"  [ABORT at step {step}] Degenerate probs after sanitize", flush=True)
                aborted = True
                break

            chosen = torch.multinomial(probs, 1)
            next_token = top_idxs[0, chosen[0, 0]].item()

            if next_token == 50256:
                break

            generated_ids.append(next_token)
            next_emb = embedding(torch.tensor([[next_token]], dtype=torch.long, device=device)).float()
            emb = torch.cat([emb, next_emb], dim=1)
            if emb.shape[1] > 64:
                emb = emb[:, -64:, :]
            
            # Print token immediately
            print(enc.decode([next_token]), end="", flush=True)

    if not aborted:
        elapsed = time.time() - t_start
        print(f"\\nTOKENS: {len(generated_ids)} | {elapsed:.1f}s | {len(generated_ids)/max(elapsed,0.01):.2f} tok/s", flush=True)

print(f"\n{'='*60}")
print("PROBE COMPLETE")
print(f"{'='*60}")
