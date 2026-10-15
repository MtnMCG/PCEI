import torch
import pytest


from src.engine.model import PCEIModel
from src.engine.ambient import AmbientField
from src.engine.seeder import DreamSeeder

def test_ambient_field_consolidation():
    from src.config import PCEIConfig
    config = PCEIConfig(d_model_text=64, layers_text=2, d_model_img=64, layers_img=1, d_model_audio=64, layers_audio=1)
    model = PCEIModel(config=config)
    field = AmbientField(model, fast_decay=0.9, slow_decay=0.99)
    
    latent1 = torch.ones(2, 5, 64)
    field.tick(latent1, loss=1.0)
    
    assert field.fast_state is not None, "Fast state should be initialized after tick"
    assert field.slow_state is not None, "Slow state should be initialized after tick"
    assert field.fast_state.shape == (2, 64)
    
    latent2 = torch.ones(2, 5, 64) * 2.0
    field.tick(latent2, loss=0.5)
    
    # Fast state should move towards 2.0
    assert torch.all(field.fast_state > 1.0), "Fast state EMA did not update toward new latent"


def test_dream_seeder_with_slow_state():
    """Test the full-spec DreamSeeder with TernaryNet dream_policy and slow_state."""
    seeder = DreamSeeder(d_model_text=256, d_model_img=512, d_model_audio=128)
    
    batch = 3
    intent = torch.randn(batch, 256)
    seq_lens = {'text': 10, 'img': 4, 'audio': 50}
    
    # Create a mock slow_state tensor
    slow_state = torch.randn(256)
    
    seeds = seeder(intent, batch, seq_lens, slow_state=slow_state, mean_confidence=0.5)
    
    assert 'text' in seeds
    assert seeds['text'].shape == (batch, 10, 256)
    assert 'img' in seeds
    assert seeds['img'].shape == (batch, 4, 512)
    assert 'audio' in seeds
    assert seeds['audio'].shape == (batch, 50, 128)

def test_dream_strength_annealing():
    """Verify dream_strength anneals toward mean_confidence over training."""
    seeder = DreamSeeder(d_model_text=256)
    seeder.train()
    
    initial_strength = seeder.dream_strength.item()
    
    intent = torch.randn(1, 256)
    seq_lens = {'text': 4}
    
    # Run many forward passes with high confidence
    for _ in range(100):
        seeder(intent, 1, seq_lens, slow_state=None, mean_confidence=0.8)
    
    final_strength = seeder.dream_strength.item()
    assert final_strength > initial_strength, \
        f"dream_strength should have annealed up: {initial_strength} → {final_strength}"
