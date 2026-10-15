"""
PCEI Diffusion Training Utilities

Shared training logic for all PCEI training scripts.
Implements proper diffusion denoising across the full noise schedule
so that training matches the Mercury-style generation used in resonance_chat.

The core insight: generation iterates from pure noise (t=T) to clean (t=0).
Training must teach the model to denoise at EVERY noise level, not just t=0.
"""

import math
import random
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def cosine_noise_schedule(t: int, T: int) -> float:
    """
    Returns the noise scale for timestep t out of T total steps.
    Uses a cosine schedule — spends more time at lower noise levels
    where fine-grained learning is most valuable.
    
    t=0       → noise_scale ≈ 0    (near-clean, easy denoising)
    t=T-1     → noise_scale ≈ 1.0  (pure noise, hardest denoising)
    """
    if T <= 1:
        return 0.0
    s = 0.008  # small offset to prevent singularity at boundaries
    f_t = math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos((s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_0
    return math.sqrt(max(1.0 - alpha_bar, 1e-8))


def add_diffusion_noise(clean_latent: torch.Tensor, t: int, T: int) -> Tuple[torch.Tensor, float]:
    """
    Adds noise to clean latents at the specified timestep level.
    
    Returns:
        noisy_latent: The noised version of the input
        noise_scale: The noise scale that was applied
    """
    noise_scale = cosine_noise_schedule(t, T)
    noise = torch.randn_like(clean_latent)
    noisy_latent = clean_latent + noise * noise_scale
    return noisy_latent, noise_scale


def diffusion_training_step(nexus, clean_latent: torch.Tensor, max_T: int = 12,
                            ce_target: Optional[torch.Tensor] = None,
                            ce_weight: float = 0.3,
                            text_latent: Optional[torch.Tensor] = None,
                            img_input: Optional[torch.Tensor] = None,
                            audio_input: Optional[torch.Tensor] = None,
                            img_latent: Optional[torch.Tensor] = None,
                            audio_latent: Optional[torch.Tensor] = None,
                            modality: str = 'text') -> Tuple[torch.Tensor, float, int, float]:
    """
    Universal PCEI diffusion training step.
    
    Samples a random timestep, adds the corresponding noise to clean_latent,
    and trains the model to recover the clean version. This teaches the model
    the full noise→signal trajectory that generation requires.
    
    Args:
        nexus: PCEINexus model (must be in train mode)
        clean_latent: (B, seq, d) clean embeddings — the denoising target
        max_T: maximum diffusion depth (should match generation --depth)
        ce_target: optional (B, seq) token IDs for cross-entropy auxiliary loss
        ce_weight: weight for CE loss (0.3 default — MSE dominates)
        text_latent: if modality != 'text', pass clean text context here
        img_input/img_latent: image context (raw patches or pre-embedded)
        audio_input/audio_latent: audio context (raw mel or pre-embedded)
        modality: which modality is being denoised ('text', 'img', 'audio')
        
    Returns:
        (total_loss, mse_value, sampled_timestep, noise_scale)
    """
    device = clean_latent.device
    
    # 1. Sample random timestep from the full range
    t = random.randint(0, max_T - 1)
    
    # 2. Add noise at this timestep level
    noisy_latent, noise_scale = add_diffusion_noise(clean_latent, t, max_T)
    
    # 3. Build forward call kwargs based on which modality we're denoising
    from src.modules.plasticity import PlasticLayer
    conf_vals = [m.slow_field_confidence.item() for m in nexus.core.modules() 
                 if isinstance(m, PlasticLayer)]
    mean_conf = sum(conf_vals) / max(len(conf_vals), 1) if conf_vals else 0.5
    
    fwd_kwargs = {
        'diffusion_timesteps': 1,
        'override_timestep': t,
        'slow_state': nexus.ambient_field.slow_state,
        'mean_confidence': mean_conf,
    }
    
    if modality == 'text':
        fwd_kwargs['text_latent'] = noisy_latent
        if img_input is not None: fwd_kwargs['img_input'] = img_input
        if img_latent is not None: fwd_kwargs['img_latent'] = img_latent
        if audio_input is not None: fwd_kwargs['audio_input'] = audio_input
        if audio_latent is not None: fwd_kwargs['audio_latent'] = audio_latent
    elif modality == 'img':
        fwd_kwargs['img_latent'] = noisy_latent
        if text_latent is not None: fwd_kwargs['text_latent'] = text_latent
        if audio_input is not None: fwd_kwargs['audio_input'] = audio_input
        if audio_latent is not None: fwd_kwargs['audio_latent'] = audio_latent
    elif modality == 'audio':
        fwd_kwargs['audio_latent'] = noisy_latent
        if text_latent is not None: fwd_kwargs['text_latent'] = text_latent
        if img_input is not None: fwd_kwargs['img_input'] = img_input
        if img_latent is not None: fwd_kwargs['img_latent'] = img_latent
    
    # 4. Forward pass
    text_logits, t_lat, i_lat, a_lat, img_recon, audio_recon = nexus.core(**fwd_kwargs)
    
    # 5. Select the denoised output for the target modality
    if modality == 'text':
        denoised = t_lat
    elif modality == 'img':
        denoised = i_lat
    elif modality == 'audio':
        denoised = a_lat
    
    # 6. MSE denoising loss (primary objective)
    mse_loss = F.mse_loss(denoised, clean_latent)
    
    # 7. Optional cross-entropy auxiliary loss (keeps lm_head trained)
    total_loss = mse_loss
    if ce_target is not None and modality == 'text':
        logits = nexus.core.text_stack.decode_output(denoised)
        ce_loss = F.cross_entropy(
            logits.transpose(1, 2), 
            ce_target, 
            ignore_index=0,
            reduction='mean'
        )
        total_loss = mse_loss + ce_loss * ce_weight
    
    return total_loss, mse_loss.item(), t, noise_scale
