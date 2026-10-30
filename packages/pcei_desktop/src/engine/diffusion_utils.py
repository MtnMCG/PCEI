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
from typing import Optional, Tuple, Union


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
    Uses the standard DDPM formulation: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    
    Returns:
        noisy_latent: The noised version of the input
        noise_scale: sqrt(1 - alpha_bar) — the noise scale that was applied
    """
    if T <= 1:
        return clean_latent, 0.0
    s = 0.008
    f_t = math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos((s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_0
    sqrt_alpha_bar = math.sqrt(max(alpha_bar, 1e-8))
    sqrt_one_minus_alpha_bar = math.sqrt(max(1.0 - alpha_bar, 1e-8))
    noise = torch.randn_like(clean_latent)
    
    # Cast scalar coefficients to tensor dtype to prevent implicit upcasting
    c_a = torch.tensor(sqrt_alpha_bar, dtype=clean_latent.dtype, device=clean_latent.device)
    c_n = torch.tensor(sqrt_one_minus_alpha_bar, dtype=clean_latent.dtype, device=clean_latent.device)
    
    noisy_latent = clean_latent * c_a + noise * c_n
    return noisy_latent, sqrt_one_minus_alpha_bar


def ddpm_reverse_step(x_t: torch.Tensor, x_0_pred: torch.Tensor, t: int, T: int) -> torch.Tensor:
    """
    Computes x_{t-1} from x_t and the predicted clean latent x_0.
    Standard DDPM sampling step.
    """
    if T <= 1 or t == 0:
        return x_0_pred
        
    s = 0.008
    # alpha_bar_t
    f_t = math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos((s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar_t = f_t / f_0
    
    # alpha_bar_{t-1}
    f_t_minus_1 = math.cos((((t - 1) / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar_t_minus_1 = f_t_minus_1 / f_0
    
    # Clip for stability
    alpha_bar_t = max(alpha_bar_t, 1e-8)
    alpha_bar_t_minus_1 = max(alpha_bar_t_minus_1, 1e-8)
    
    alpha_t = alpha_bar_t / alpha_bar_t_minus_1
    beta_t = 1.0 - alpha_t
    
    # Posterior mean computation
    coeff1 = math.sqrt(alpha_bar_t_minus_1) * beta_t / (1.0 - alpha_bar_t)
    coeff2 = math.sqrt(alpha_t) * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t)
    
    c1 = torch.tensor(coeff1, dtype=x_t.dtype, device=x_t.device)
    c2 = torch.tensor(coeff2, dtype=x_t.dtype, device=x_t.device)
    
    posterior_mean = c1 * x_0_pred + c2 * x_t
    
    # Posterior variance
    posterior_variance = max(beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t), 1e-8)
    pv = torch.tensor(math.sqrt(posterior_variance), dtype=x_t.dtype, device=x_t.device)
    
    noise = torch.randn_like(x_t)
    return posterior_mean + pv * noise


def compute_snr(t: Union[int, float, torch.Tensor], T: int) -> Union[float, torch.Tensor]:
    """
    Computes the Signal-to-Noise Ratio (SNR) for timestep t out of T total steps
    under the cosine noise schedule:
        SNR(t) = alpha_t / sigma_t^2
    where:
        alpha_t = alpha_bar_t  (signal variance)
        sigma_t^2 = 1 - alpha_bar_t  (noise variance)
    """
    if isinstance(t, torch.Tensor):
        if T <= 1:
            return torch.full_like(t, 1e5, dtype=torch.float32)
        s = 0.008
        t_f = t.float()
        f_t = torch.cos(((t_f / T) + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos((s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_0
        alpha_t = torch.clamp(alpha_bar, min=1e-8, max=1.0 - 1e-8)
        sigma_t_sq = torch.clamp(1.0 - alpha_t, min=1e-8)
        return alpha_t / sigma_t_sq
    else:
        if T <= 1:
            return 1e5
        s = 0.008
        f_t = math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos((s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_0
        alpha_t = max(min(alpha_bar, 1.0 - 1e-8), 1e-8)
        sigma_t_sq = max(1.0 - alpha_t, 1e-8)
        return alpha_t / sigma_t_sq


def compute_min_snr_weight(t: Union[int, float, torch.Tensor], T: int, max_snr: float = 5.0) -> Union[float, torch.Tensor]:
    """
    Computes Min-SNR loss weighting (Hang et al., 2023):
        SNR(t) = alpha_t / sigma_t^2
        w_t = min(SNR(t), max_snr) / SNR(t)
    
    For clean latent x_0 prediction, this prevents trivial near-zero noise steps
    from dominating gradient updates while maintaining full weighting on high-noise steps.
    """
    snr = compute_snr(t, T)
    if isinstance(snr, torch.Tensor):
        clamped_snr = torch.clamp(snr, min=1e-8)
        min_snr = torch.clamp(snr, max=max_snr)
        return min_snr / clamped_snr
    else:
        if snr <= 1e-8:
            return 1.0
        return min(snr, max_snr) / snr


def diffusion_training_step(nexus, clean_latent: torch.Tensor, max_T: int = 12,
                            ce_target: Optional[torch.Tensor] = None,
                            ce_weight: float = 0.3,
                            text_latent: Optional[torch.Tensor] = None,
                            img_input: Optional[torch.Tensor] = None,
                            audio_input: Optional[torch.Tensor] = None,
                            img_latent: Optional[torch.Tensor] = None,
                            audio_latent: Optional[torch.Tensor] = None,
                            modality: str = 'text',
                            use_min_snr: bool = False,
                            min_snr_gamma: float = 5.0,
                            labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, int, float]:
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
    device = clean_latent.device if clean_latent is not None else (ce_target.device if ce_target is not None else torch.device('cpu'))
    
    # 1. Sample random timestep from the full range (for non-text modalities)
    t = random.randint(0, max_T - 1)
    
    # 2. Add noise (Skip for text since we are pivoting to autoregressive)
    if modality != 'text':
        noisy_latent, noise_scale = add_diffusion_noise(clean_latent, t, max_T)
    else:
        noisy_latent = clean_latent
        noise_scale = 0.0
        t = 0  # Autoregressive doesn't use diffusion timesteps
        
    # 3. Build forward call kwargs based on which modality we're training
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
        if ce_target is not None:
            fwd_kwargs['text_input'] = ce_target
        else:
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
    
    # 5. Modality-specific loss
    if modality == 'text':
        # --- AUTOREGRESSIVE TEXT PIVOT ---
        # Decode the full sequence from Mamba (reuse text_logits from forward pass to save memory)
        logits = text_logits if text_logits is not None else nexus.core.text_stack.decode_output(t_lat)
        
        if ce_target is None:
            # Dynamically reconstruct target tokens from clean_latent (self-supervised target)
            with torch.no_grad():
                target_logits = nexus.core.text_stack.decode_output(clean_latent)
                ce_target = torch.argmax(target_logits, dim=-1)
            
        # Shift targets for Next-Token Prediction
        shift_logits = logits[:, :-1, :].contiguous()
        target_seq = labels if labels is not None else ce_target
        shift_labels = target_seq[:, 1:].contiguous()
        
        # --- CHUNKED CROSS-ENTROPY ---
        # The monolithic F.cross_entropy needs to materialize (1, vocab_size, seq_len)
        # which is ~73MB at 768 tokens and triggers OOM on MPS every time.
        # Chunking to 128 tokens keeps peak allocation under 12MB.
        chunk_size = 128
        seq_len = shift_logits.size(1)
        total_ce = shift_logits.sum() * 0.0
        total_tokens = 0
        for c_start in range(0, seq_len, chunk_size):
            c_end = min(c_start + chunk_size, seq_len)
            chunk_logits = shift_logits[:, c_start:c_end, :]
            chunk_labels = shift_labels[:, c_start:c_end]
            # Count non-padding tokens in this chunk for proper weighted average
            valid = (chunk_labels != 0).sum().item()
            if valid == 0:
                continue
            chunk_loss = F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=0,
                reduction='sum'
            )
            total_ce = total_ce + chunk_loss
            total_tokens += valid
        
        ce_loss = total_ce / max(total_tokens, 1)
        del shift_logits, shift_labels  # Free immediately
        
        # Return identically formatted tuple so downstream scripts don't break
        return ce_loss, ce_loss.item(), 0, 0.0
        
    elif modality == 'img':
        denoised = i_lat
        mse_loss = F.mse_loss(denoised, clean_latent)
        if use_min_snr:
            w_t = compute_min_snr_weight(t, max_T, max_snr=min_snr_gamma)
            return w_t * mse_loss, mse_loss.item(), t, noise_scale
        return mse_loss, mse_loss.item(), t, noise_scale
        
    elif modality == 'audio':
        denoised = a_lat
        mse_loss = F.mse_loss(denoised, clean_latent)
        if use_min_snr:
            w_t = compute_min_snr_weight(t, max_T, max_snr=min_snr_gamma)
            return w_t * mse_loss, mse_loss.item(), t, noise_scale
        return mse_loss, mse_loss.item(), t, noise_scale

    elif modality in ('diffusion_text', 'latent'):
        denoised = t_lat
        mse_loss = F.mse_loss(denoised, clean_latent)
        if use_min_snr:
            w_t = compute_min_snr_weight(t, max_T, max_snr=min_snr_gamma)
            return w_t * mse_loss, mse_loss.item(), t, noise_scale
        return mse_loss, mse_loss.item(), t, noise_scale
