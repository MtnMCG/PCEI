"""
Test: Training Convergence Smoke Test
Verifies that the full training loop can reduce loss over a small number of steps.
"""
import torch
import torch.nn as nn
import pytest

from src.pcei import PCEINexus
from src.config import PCEIConfig


def test_training_loss_decreases():
    """Smoke test: 30 steps of training on a fixed batch should reduce loss."""
    config = PCEIConfig(
        vocab_size=1000, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=2, layers_img=1, layers_audio=1
    )
    nexus = PCEINexus(config=config)
    nexus.train()
    
    optimizer = torch.optim.AdamW(nexus.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    tokens = torch.randint(1, 1000, (2, 32))
    losses = []
    for step in range(30):
        logits, t_lat, i_lat, a_lat, img_r, aud_r = nexus.core(
            text_input=tokens[:, :-1],
            img_input=None,
            audio_input=None,
            diffusion_timesteps=1,
            slow_state=nexus.ambient_field.slow_state,
            mean_confidence=0.5
        )
        
        if logits is None:
            continue
            
        loss = criterion(
            logits[:, :31, :].reshape(-1, 1000),
            tokens[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nexus.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    
    assert len(losses) > 0, "No losses were recorded"
    first_half_avg = sum(losses[:5]) / 5
    second_half_avg = sum(losses[-5:]) / 5
    assert second_half_avg < first_half_avg, \
        f"Loss did not decrease: first half avg {first_half_avg:.2f} → second half avg {second_half_avg:.2f}"

