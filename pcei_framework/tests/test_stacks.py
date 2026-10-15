import torch
import pytest


from src.modules.plasticity import PlasticLayer
from src.core.stack import TextStack, ImageStack, AudioStack

def test_plastic_layer_lr_modulation():
    layer = PlasticLayer(layer_id="test_layer", d_model=128, initial_pos=(1.0, 2.0, 3.0), base_lr=1e-3)
    
    # Check default effective learning rate
    assert layer.effective_lr == 1e-3
    
    # Increase slow field confidence to 0.75
    layer.slow_field_confidence.copy_(torch.tensor([0.75]))
    
    # The effective LR should be scaled by (1.0 - 0.75) = 0.25 -> 2.5e-4
    assert abs(layer.effective_lr - 2.5e-4) < 1e-6
    
def test_plastic_layer_position():
    layer = PlasticLayer(layer_id="test_pos", d_model=128, initial_pos=(0.0, 0.0, 5.0))
    pos = layer.get_position()
    assert pos == (0.0, 0.0, 5.0)
    
    layer.update_position(0.5, -0.2)
    pos = layer.get_position()
    assert abs(pos[0] - 0.5) < 1e-6
    assert abs(pos[1] - (-0.2)) < 1e-6
    assert abs(pos[2] - 5.0) < 1e-6

def test_text_stack_forward():
    stack = TextStack(vocab_size=1000, d_model=128, num_layers=2)
    
    # Simulate a batch of 2, seq_len 10 tokens
    tokens = torch.randint(0, 1000, (2, 10))
    time = torch.zeros(2) # Timestep 0
    
    # Embed
    latents = stack.integrate_input(tokens)
    assert latents.shape == (2, 10, 128)
    
    # Diffuse physically
    denoised = stack(latents, time)
    assert denoised.shape == (2, 10, 128)
    
    # Decode
    logits = stack.decode_output(denoised)
    assert logits.shape == (2, 10, 1000)

def test_image_and_audio_stacks_forward():
    img_stack = ImageStack(d_model=256, num_layers=2)
    audio_stack = AudioStack(d_model=128, num_layers=2)
    
    # Image dummy forward
    img_x = torch.randn(2, 64, 256) # batch 2, 64 patches, d_model 256
    img_t = torch.zeros(2)
    img_out = img_stack(img_x, img_t)
    assert img_out.shape == img_x.shape
    
    # Audio dummy forward
    aud_x = torch.randn(2, 100, 128) # batch 2, 100 frames, d_model 128
    aud_t = torch.zeros(2)
    aud_out = audio_stack(aud_x, aud_t)
    assert aud_out.shape == aud_x.shape
