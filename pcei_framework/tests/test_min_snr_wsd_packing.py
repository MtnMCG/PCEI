import math
import torch
import torch.nn as nn
import pytest

from src.engine.diffusion_utils import (
    compute_snr,
    compute_min_snr_weight,
    diffusion_training_step,
    add_diffusion_noise,
    cosine_noise_schedule,
)
from scripts.train import PCEITokenizer, ContinuousSequencePacker
from src.pcei import PCEINexus
from src.config import PCEIConfig


def test_min_snr_calculation():
    """Verify Min-SNR calculation SNR(t) = alpha_t / sigma_t^2 and w_t = min(SNR(t), 5.0) / SNR(t)."""
    T = 12
    # Verify scalar inputs across all timesteps
    weights = []
    for t in range(T):
        snr = compute_snr(t, T)
        w_t = compute_min_snr_weight(t, T, max_snr=5.0)
        assert not math.isnan(snr) and not math.isinf(snr), f"SNR at t={t} is NaN or Inf"
        assert not math.isnan(w_t) and not math.isinf(w_t), f"w_t at t={t} is NaN or Inf"
        assert 0.0 <= w_t <= 1.0, f"w_t at t={t} should be in [0, 1], got {w_t}"
        weights.append(w_t)

    # At low noise (t=1), SNR should be high, so w_t < 1.0
    assert compute_snr(1, T) > 5.0
    assert compute_min_snr_weight(1, T, max_snr=5.0) < 1.0

    # At high noise (e.g. t=6), SNR should be low (< 5.0), so w_t == 1.0
    assert compute_snr(6, T) < 5.0
    assert compute_min_snr_weight(6, T, max_snr=5.0) == 1.0

    # Verify tensor vectorization
    t_tensor = torch.arange(T)
    w_tensor = compute_min_snr_weight(t_tensor, T, max_snr=5.0)
    assert w_tensor.shape == (T,)
    assert not torch.isnan(w_tensor).any()
    assert not torch.isinf(w_tensor).any()
    assert torch.all(w_tensor <= 1.0)
    assert torch.all(w_tensor >= 0.0)


def test_continuous_sequence_packer():
    """Verify that continuous sequence packing packs documents separated by <|endoftext|> with NO padding zeros."""
    tokenizer = PCEITokenizer(vocab_size=50257)
    sample_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Differential geometry and topological manifold learning.",
        "Mercury-style diffusion generation over latent manifolds.",
        "Continuous sequence packing ensures 100% compute efficiency on Apple Silicon.",
    ]
    max_len = 64
    batch_size = 2
    packer = ContinuousSequencePacker(sample_docs, tokenizer, max_seq_len=max_len)

    # Fetch multiple batches to test stream replenishment
    for _ in range(3):
        batch = packer.get_batch(batch_size=batch_size, device="cpu")
        tokens = batch['text']
        assert tokens.shape == (batch_size, max_len)
        # CRITICAL: 100% of tokens must be real information tokens — NO zeros (pad_id)
        zero_count = (tokens == tokenizer.pad_id).sum().item()
        assert zero_count == 0, f"Found {zero_count} zero tokens; sequence packing must contain 0 padding zeros!"
        # Check that EOS tokens (50256) are present to separate documents
        assert (tokens == tokenizer.eos_id).any(), "Expected <|endoftext|> delimiter in packed sequence stream."


def test_wsd_scheduler():
    """Verify WSD scheduling: 5% warmup, 80% stable max LR, 15% cosine decay."""
    from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR, SequentialLR

    param = nn.Parameter(torch.zeros(1))
    base_lr = 1e-3
    optimizer = torch.optim.AdamW([param], lr=base_lr)
    total_steps = 100

    warmup_steps = max(1, int(round(total_steps * 0.05)))   # 5
    stable_steps = max(1, int(round(total_steps * 0.80)))   # 80
    decay_steps = max(1, total_steps - warmup_steps - stable_steps) # 15

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    stable_sched = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
    decay_sched = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, stable_sched, decay_sched],
        milestones=[warmup_steps, warmup_steps + stable_steps]
    )

    lrs = []
    for s in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Step 0: warmup start (0.01 * base_lr)
    assert math.isclose(lrs[0], 0.01 * base_lr, rel_tol=1e-3)
    # Step 5 (end of warmup): should hit base_lr
    assert math.isclose(lrs[warmup_steps], base_lr, rel_tol=1e-3)
    # Stable phase (steps 5 to 84): exactly base_lr
    for s in range(warmup_steps, warmup_steps + stable_steps):
        assert math.isclose(lrs[s], base_lr, rel_tol=1e-4)
    # Decay phase starts at step 85
    assert lrs[85] == base_lr
    # Decay phase middle at step 92: LR must be less than base_lr
    assert lrs[92] < base_lr
    # Step 99: Decayed near eta_min (1e-6)
    assert lrs[-1] < 0.05 * base_lr


def test_synthetic_forward_backward_pass():
    """Small synthetic forward/backward pass dry run to ensure no NaNs or broken backward graph."""
    config = PCEIConfig.load()
    # Use compact topology for fast dry run
    config.layers_text = 2
    config.layers_img = 1
    config.layers_audio = 1
    config.d_model_text = 128
    config.d_model_img = 64
    config.d_model_audio = 64

    nexus = PCEINexus(config=config, device="cpu")
    nexus.train()
    dtype = nexus.system_dtype

    # 1. Forward/backward with Min-SNR on text
    clean_text_latent = torch.randn(2, 16, config.d_model_text, dtype=dtype, requires_grad=True)
    tokens = torch.randint(1, 1000, (2, 16))
    loss, mse_val, t, scale = diffusion_training_step(
        nexus, clean_text_latent, max_T=6,
        ce_target=tokens, ce_weight=0.3,
        use_min_snr=True
    )
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    loss.backward()

    # Zero grads between passes
    nexus.zero_grad()

    # 2. Forward/backward with Min-SNR on image modality (MSE denoising)
    clean_img_latent = torch.randn(2, 8, config.d_model_img, dtype=dtype, requires_grad=True)
    text_ctx = torch.randn(2, 16, config.d_model_text, dtype=dtype)
    img_loss, mse_val, t, scale = diffusion_training_step(
        nexus, clean_img_latent, max_T=6,
        text_latent=text_ctx,
        modality='img',
        use_min_snr=True
    )
    assert not torch.isnan(img_loss), "Image loss is NaN"
    assert not torch.isinf(img_loss), "Image loss is Inf"
    img_loss.backward()

    # 3. Check parameter gradients have no NaNs
    has_grads = False
    for name, param in nexus.named_parameters():
        if param.grad is not None:
            has_grads = True
            assert not torch.isnan(param.grad).any(), f"Param {name} gradient contains NaN"
            assert not torch.isinf(param.grad).any(), f"Param {name} gradient contains Inf"
    assert has_grads, "No gradients were computed!"
