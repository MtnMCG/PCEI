import torch
import pytest


from src.pcei import PCEINexus
from src.config import PCEIConfig

def test_pcei_nexus_forward():
    config = PCEIConfig(vocab_size=100, d_model_text=32, layers_text=2, d_model_img=32, layers_img=1, d_model_audio=32, layers_audio=1)
    nexus = PCEINexus(config=config)
    nexus.train()
    
    batch = 2
    text = torch.randint(0, 100, (batch, 128))
    img = torch.randn(batch, 3, 32, 32)
    aud = torch.randn(batch, 100, 80)
    
    nexus.response_gate.reset()
    _ = nexus(text, img, aud, max_thinking_steps=2)
    logits_active = nexus(text, img, aud, max_thinking_steps=2)
    
    if logits_active is not None:
        assert logits_active.shape == (batch, 128, 100)
    
def test_pcei_nexus_witness_mode():
    config = PCEIConfig(vocab_size=100, d_model_text=32, layers_text=2, d_model_img=32, layers_img=1, d_model_audio=32, layers_audio=1)
    nexus = PCEINexus(config=config)
    nexus.set_mode("WITNESS")
    
    batch = 2
    text = torch.randint(0, 100, (batch, 128))
    img = torch.randn(batch, 3, 32, 32)
    
    logits_witness = nexus(text, img, audio_tensor=None, max_thinking_steps=2)
    
    assert logits_witness is None, "Nexus produced output while in WITNESS mode"

def test_pcei_nexus_consolidation_integration():
    config = PCEIConfig(d_model_text=32, layers_text=2, d_model_img=32, layers_img=1, d_model_audio=32, layers_audio=1)
    nexus = PCEINexus(config=config)
    
    t_latent = torch.randn(2, 5, 32)
    nexus.ambient_field.tick(t_latent, loss=0.5)
    
    assert nexus.ambient_field.fast_state is not None, "Ambient field fast state failed to initialize"
    assert len(nexus.core.text_stack.layers) > 0, "Core model missing stacks"
    first_layer = nexus.core.text_stack.layers[0]
    if hasattr(first_layer, 'module'):
        first_layer = first_layer.module
    assert hasattr(first_layer.block.mamba, "conv1d"), "Mamba sequence backbone missing"


