import torch
import pytest


from src.engine.model import PCEIModel
from src.config import PCEIConfig

def test_pcei_multimodal_forward():
    config = PCEIConfig(
        vocab_size=100,
        d_model_text=64, layers_text=2,
        d_model_img=64, layers_img=1,
        d_model_audio=64, layers_audio=1
    )
    model = PCEIModel(config=config)
    
    batch = 2
    text_input = torch.randint(0, 100, (batch, 128))
    img_input = torch.randn(batch, 3, 32, 32)
    audio_input = torch.randn(batch, 100, 80)
    
    text_logits, t_latent, i_latent, a_latent, img_recon, audio_recon = model(
        text_input=text_input,
        img_input=img_input,
        audio_input=audio_input,
        diffusion_timesteps=2
    )
    
    assert text_logits is not None, "Text logits should be generated"
    assert text_logits.shape == (batch, 128, 100), f"Expected shape (2, 128, 100), got {text_logits.shape}"
    
    assert i_latent is not None, "Image latent should be retained"
    assert i_latent.shape == (batch, 4, 64), f"Expected shape (2, 4, 64), got {i_latent.shape}"
    
    assert a_latent is not None, "Audio latent should be retained"
    assert a_latent.shape == (batch, 100, 64), f"Expected shape (2, 100, 64), got {a_latent.shape}"
    
    assert img_recon is not None, "Image reconstruction should be present"
    assert audio_recon is not None, "Audio reconstruction should be present"

