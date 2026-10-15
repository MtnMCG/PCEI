import torch
import torch.nn.functional as F
from pathlib import Path
from src.pcei import PCEINexus
import tiktoken

device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

print("Loading PCEI 750M Checkpoint...")
ckpt_path = Path("checkpoints/pcei_100m.pt")
layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint(ckpt_path)

nexus = PCEINexus(
    vocab_size=50257, device=device,
    d_model_text=d_text, d_model_img=d_img, d_model_audio=d_audio,
    layers_text=layers_text, layers_img=layers_img, layers_audio=layers_audio
)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
nexus.load_state_dict(ckpt['model_state_dict'], strict=False)
nexus.to(nexus.system_dtype)
nexus.eval()

def evaluate_knowledge(text, name):
    ids = enc.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Get clean latents
        clean_latent = nexus.core.text_stack.integrate_input(input_ids)
        
        # Test 1: Autoregressive Next-Token Predictability (t=0, clean context)
        logits, _, _, _, _, _ = nexus.core(
            text_latent=clean_latent,
            diffusion_timesteps=1, override_timestep=0,
            slow_state=nexus.ambient_field.slow_state,
            mean_confidence=0.5
        )
        
        logits = nexus.core.text_stack.decode_output(clean_latent)
        
        # Shift logits and targets for next-token prediction
        shift_logits = logits[0, :-1, :].float()
        shift_labels = input_ids[0, 1:]
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        print(f"[{name}] Cross-Entropy Loss: {loss.item():.4f}")

print("\n--- Probing Underlying Knowledge ---")
# 1. Real, coherent knowledge
text1 = "SYSTEM: You are a trading assistant.\nUSER: What is the impact of rising interest rates?\nASSISTANT: Rising interest rates generally cause bond prices to fall and can cool down economic growth."
evaluate_knowledge(text1, "Coherent Finance")

# 2. Random gibberish
import random
random_ids = [random.randint(0, 50000) for _ in range(50)]
text2 = enc.decode(random_ids)
evaluate_knowledge(text2, "Random Gibberish")

# 3. Simple English
text3 = "Hello, how are you doing today? I am doing well, thank you."
evaluate_knowledge(text3, "Simple English")

