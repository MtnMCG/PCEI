import torch
import torch.nn as nn
from typing import List, Dict


from src.modules.plasticity import PlasticLayer
from src.modules.adaptation import TTTAdapter

class ModalityStack(nn.Module):
    """
    Base class for a modality processing column.
    A Stack represents a sequence of processing layers (PlasticLayers) 
    that denoise latent embeddings bounded to a specific modality.
    """
    def __init__(self, stack_id: str, d_model: int, num_layers: int, start_z: float):
        super().__init__()
        self.stack_id = stack_id
        self.d_model = d_model
        
        # Build the sequence of strictly local, continuous reasoning blocks
        # Initial X,Y positions for layers within a stack usually start stacked at 0,0,
        # but Z increments representing depth into the model's core.
        layers = []
        for i in range(num_layers):
            layer_z = start_z + float(i)
            layer_name = f"{stack_id}_L{i}"
            
            # Add a tiny "Big Bang" jitter to break collinear symmetry
            jitter_x = (torch.rand(1).item() - 0.5) * 1e-3
            jitter_y = (torch.rand(1).item() - 0.5) * 1e-3
            
            layers.append(
                PlasticLayer(layer_id=layer_name, d_model=d_model, initial_pos=(jitter_x, jitter_y, layer_z))
            )
        
        # Wrap the last layer in TTTAdapter for test-time adaptation (spec Milestone 4)
        if len(layers) > 0:
            last_layer = layers[-1]
            layers[-1] = TTTAdapter(module=last_layer, lr=1e-4, num_steps=1)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, time: torch.Tensor, detectors: Dict[str, nn.Module] = None) -> torch.Tensor:
        """
        Passes the latent embedding through the physical stack.
        (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """
        for layer in self.layers:
            if isinstance(layer, TTTAdapter):
                x = layer(x, time, is_training=self.training)
            else:
                x = layer(x, time)
            # Universal Registration: Every layer in every stack can now resonate
            # For TTTAdapter-wrapped layers, get the inner layer_id
            inner = layer.module if isinstance(layer, TTTAdapter) else layer
            if detectors and hasattr(inner, 'layer_id') and inner.layer_id in detectors:
                detectors[inner.layer_id].register_activation(x.detach())
        return x

class TextStack(ModalityStack):
    """
    Stack A (Logic/Text). 
    Asymmetric depth: Deeper than sensory stacks to process complex relationship reasoning.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, num_layers: int = 12, start_z: float = 0.0):
        # We start the Text Stack at Z=0.0 moving positive
        super().__init__(stack_id="TEXT", d_model=d_model, num_layers=num_layers, start_z=start_z)
        self.vocab_size = vocab_size
        
        # Tokenizer entrance bounding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Detokenizer exit bounding
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def integrate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Translates discrete tokens into the continuous embedding space."""
        return self.embedding(input_ids)

    def decode_output(self, latents: torch.Tensor) -> torch.Tensor:
        """Translates continuous latents back to distinct logits."""
        return self.lm_head(self.norm_final(latents))

class ImageStack(ModalityStack):
    """
    Stack B (Visual). 
    Wider and shallower than text. Focuses purely on topological pattern matching.
    """
    def __init__(self, d_model: int = 512, num_layers: int = 6, start_z: float = 20.0):
        # We start the Image Stack far down the Z axis, e.g. Z=20.0
        super().__init__(stack_id="IMAGE", d_model=d_model, num_layers=num_layers, start_z=start_z)
        
        # A lightweight patch embedding layer (e.g. projecting 16x16 patches)
        # simplified for prototype
        self.patch_dim = 16 * 16 * 3
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.norm = nn.LayerNorm(d_model) # Stabilize multimodal injection
        
        # Reconstruction decoder: latent → reconstructed patches (Phase 3 / Milestone 11)
        self.patch_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, self.patch_dim),
            nn.Sigmoid()  # Pixel values in [0, 1]
        )
        
    def integrate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Projects raw patches and normalizes the latent space."""
        return self.norm(self.patch_proj(x))

    def decode_output(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstructs raw patches from latent space."""
        return self.patch_decoder(latents)
        
class AudioStack(ModalityStack):
    """
    Stack C (Acoustic). 
    Focuses on temporal waveform representations.
    """
    def __init__(self, d_model: int = 256, num_layers: int = 4, start_z: float = 40.0):
        # Starts even further down Z axis
        super().__init__(stack_id="AUDIO", d_model=d_model, num_layers=num_layers, start_z=start_z)
        
        # Mel spectrogram embedding layer
        self.mel_bins = 80
        self.audio_proj = nn.Linear(self.mel_bins, d_model)
        self.norm = nn.LayerNorm(d_model) # Stabilize multimodal injection
        
        # Reconstruction decoder: latent → reconstructed mel spectrogram (Phase 3 / Milestone 11)
        self.mel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, self.mel_bins),
        )

    def integrate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Projects mel spectrograms and normalizes the latent space."""
        return self.norm(self.audio_proj(x))

    def decode_output(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstructs mel spectrogram frames from latent space."""
        return self.mel_decoder(latents)

