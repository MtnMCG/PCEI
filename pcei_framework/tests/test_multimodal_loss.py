"""
Test: Multimodal Reconstruction Loss — verify gradient flow to image/audio stacks
"""
import torch
import torch.nn as nn
import pytest

from src.engine.model import PCEIModel
from src.config import PCEIConfig


def test_image_reconstruction_gradient_flow():
    """Verify image reconstruction loss produces gradients in ImageStack decoder."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=2, layers_img=2, layers_audio=1
    )
    model = PCEIModel(config=config)
    model.train()
    
    batch = 2
    img = torch.randn(batch, 3, 32, 32)
    text = torch.randint(0, 100, (batch, 16))
    
    logits, t_lat, i_lat, a_lat, img_recon, audio_recon = model(
        text_input=text, img_input=img, audio_input=None,
        diffusion_timesteps=1,
        slow_state=None, mean_confidence=0.5
    )
    
    assert img_recon is not None, "ImageStack did not produce reconstruction output"
    
    target_img = img.view(batch, -1, model.img_stack.patch_dim)
    loss = nn.functional.mse_loss(img_recon, target_img)
    loss.backward()
    
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.img_stack.patch_decoder.parameters()
    )
    assert has_grad, "ImageStack decoder received no gradients from reconstruction loss"


def test_audio_reconstruction_gradient_flow():
    """Verify audio reconstruction loss produces gradients in AudioStack decoder."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=2, layers_img=1, layers_audio=2
    )
    model = PCEIModel(config=config)
    model.train()
    
    batch = 2
    audio = torch.randn(batch, 100, 80)
    text = torch.randint(0, 100, (batch, 16))
    
    logits, t_lat, i_lat, a_lat, img_recon, audio_recon = model(
        text_input=text, img_input=None, audio_input=audio,
        diffusion_timesteps=1,
        slow_state=None, mean_confidence=0.5
    )
    
    assert audio_recon is not None, "AudioStack did not produce reconstruction output"
    
    loss = nn.functional.mse_loss(audio_recon, audio)
    loss.backward()
    
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.audio_stack.mel_decoder.parameters()
    )
    assert has_grad, "AudioStack decoder received no gradients from reconstruction loss"


def test_model_returns_six_values():
    """Verify the new 6-value return signature."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=1, layers_img=1, layers_audio=1
    )
    model = PCEIModel(config=config)
    
    text = torch.randint(0, 100, (2, 16))
    img = torch.randn(2, 3, 32, 32)
    audio = torch.randn(2, 100, 80)
    
    result = model(
        text_input=text, img_input=img, audio_input=audio,
        diffusion_timesteps=1, slow_state=None, mean_confidence=0.5
    )
    
    assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
    logits, t_lat, i_lat, a_lat, img_recon, audio_recon = result
    
    assert logits is not None, "text_logits should not be None when text was provided"
    assert img_recon is not None, "img_recon should not be None when image was provided"
    assert audio_recon is not None, "audio_recon should not be None when audio was provided"

