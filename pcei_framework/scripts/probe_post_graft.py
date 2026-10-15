import time
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.pcei import PCEINexus
from src.config import PCEIConfig
from src.modules.plasticity import PlasticLayer
from src.modules.adaptation import TTTAdapter
import tiktoken

def run_probe(ckpt_filename="pcei_750m_grafted.pt"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")
    
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / ckpt_filename
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return
        
    print(f"\n{'='*70}")
    print(f" PROBING CHECKPOINT: {ckpt_filename} on {device.upper()}")
    print(f"{'='*70}")
    
    t0 = time.time()
    layers_t, layers_i, layers_a, dt, di, da = PCEINexus.load_topology_from_checkpoint(
        ckpt_path, default_text=24, default_img=4, default_audio=4,
        def_dt=1280, def_di=512, def_da=256
    )
    print(f"Topology: {layers_t}L Text ({dt}D) | {layers_i}L Img ({di}D) | {layers_a}L Audio ({da}D)")
    
    config = PCEIConfig(
        d_model_text=dt, d_model_img=di, d_model_audio=da,
        layers_text=layers_t, layers_img=layers_i, layers_audio=layers_a,
        vocab_size=50257
    )
    nexus = PCEINexus(config=config, device=device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    nexus.load_state_dict(state_dict, strict=False)
    nexus.to(nexus.system_dtype)
    nexus.eval()
    print(f"Loaded {ckpt_filename} in {time.time() - t0:.2f}s (Global Step: {ckpt.get('global_step', 'N/A')})")
    
    # 1. Parameter Sanity
    nan_params = sum(torch.isnan(p).sum().item() for p in nexus.parameters())
    inf_params = sum(torch.isinf(p).sum().item() for p in nexus.parameters())
    print(f"Integrity Scan: {nan_params} NaNs | {inf_params} Infs across parameters.")
    
    # 2. Perplexity / Cross-Entropy Loss Tests
    print(f"\n--- Cross-Entropy Loss Evaluation ---")
    test_cases = [
        ("Coherent English", "The quick brown fox jumps over the lazy dog in the warm summer sunlight."),
        ("Formal Reasoning", "If all humans are mortal, and Socrates is a human, then Socrates must be mortal."),
        ("Python Code", "def add_numbers(a, b):\n    '''Returns the sum of two integers.'''\n    return a + b\n"),
        ("Random Noise", "xkq91 zvm! @@#9921 aalsdkjf 98314 plmqwoeiru zxcvbnm"),
    ]
    
    embedding = nexus.core.text_stack.embedding
    lm_head = nexus.core.text_stack.lm_head
    norm_final = nexus.core.text_stack.norm_final
    
    with torch.no_grad():
        for label, text in test_cases:
            ids = enc.encode(text, allowed_special={"<|endoftext|>"})
            if len(ids) < 2:
                continue
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            latents = embedding(input_tensor).to(nexus.system_dtype)
            
            # Pass through text stack with timestep 0 (pure inference)
            t_batch = torch.zeros(latents.shape[0], device=device, dtype=nexus.system_dtype)
            for layer in nexus.core.text_stack.layers:
                if isinstance(layer, TTTAdapter):
                    latents = layer(latents, t_batch, is_training=False)
                else:
                    latents = layer(latents, t_batch)
                    
            normed = norm_final(latents)
            logits = lm_head(normed).float()
            
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = input_tensor[0, 1:].contiguous()
            loss = F.cross_entropy(shift_logits, shift_labels).item()
            print(f"  [{label:18s}] Cross-Entropy Loss: {loss:.4f} (Perplexity: {min(torch.exp(torch.tensor(loss)).item(), 99999.0):.1f})")
            
    # 3. Autoregressive Text Generation
    print(f"\n--- Autoregressive Generation Probe ---")
    prompts = [
        "The capital of France is",
        "In Python, to define a function you write",
        "Hello, my name is",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        raw_ids = enc.encode(prompt)
        curr_ids = list(raw_ids)
        generated = []
        
        with torch.no_grad():
            for step in range(20):
                in_tensor = torch.tensor([curr_ids[-32:]], dtype=torch.long, device=device)
                latents = embedding(in_tensor).to(nexus.system_dtype)
                t_batch = torch.zeros(1, device=device, dtype=nexus.system_dtype)
                
                for layer in nexus.core.text_stack.layers:
                    if isinstance(layer, TTTAdapter):
                        latents = layer(latents, t_batch, is_training=False)
                    else:
                        latents = layer(latents, t_batch)
                        
                normed = norm_final(latents[:, -1:, :])
                next_logits = lm_head(normed).squeeze().float()
                
                # Temperature scaling & Top-K sampling
                temp = 0.7
                scaled_logits = next_logits / temp
                top_k = 40
                top_vals, top_idxs = torch.topk(scaled_logits, top_k)
                probs = torch.softmax(top_vals, dim=-1)
                
                chosen_idx = torch.multinomial(probs, 1).item()
                next_token = top_idxs[chosen_idx].item()
                
                if next_token == 50256: # EOS
                    break
                    
                curr_ids.append(next_token)
                generated.append(next_token)
                
        gen_text = enc.decode(generated)
        print(f"Generation: '{gen_text}'")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "pcei_750m_grafted.pt"
    run_probe(ckpt)
