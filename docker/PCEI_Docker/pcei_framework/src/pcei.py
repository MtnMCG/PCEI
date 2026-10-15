import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path

from src.engine.model import PCEIModel
from src.engine.autonomy import ResponseGate, OutputRouter
from src.engine.ambient import AmbientField
from src.engine.neurogenesis import NeurogenesisController

class PCEINexus(nn.Module):
    """
    The highest-level integration interface for the PCEI Framework.
    Wraps the core Physics Engine (PCEIModel) with the Autonomy, 
    Consolidation, and I/O Routing systems.
    
    Expected Inputs:
        - text_tokens: (batch, seq_len) discrete tokens
        - image_tensor: (batch, C, H, W) raw image pixels
        - audio_tensor: (batch, time, mel_bins) or raw waveform
    """
    @classmethod
    def load_topology_from_checkpoint(cls, ckpt_path: Path, default_text=2, default_img=1, default_audio=1, def_dt=128, def_di=128, def_da=64):
        if not ckpt_path.exists():
            return default_text, default_img, default_audio, def_dt, def_di, def_da
            
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            t_layers, i_layers, a_layers = set(), set(), set()
            dt, di, da = def_dt, def_di, def_da
            
            for key in state_dict.keys():
                if "core.text_stack.layers." in key:
                    t_layers.add(key.split("core.text_stack.layers.")[1].split(".")[0])
                elif "core.img_stack.layers." in key:
                    i_layers.add(key.split("core.img_stack.layers.")[1].split(".")[0])
                elif "core.audio_stack.layers." in key:
                    a_layers.add(key.split("core.audio_stack.layers.")[1].split(".")[0])
                    
                if "core.text_stack.layers.0.block.norm1.weight" in key: dt = state_dict[key].shape[0]
                elif "core.img_stack.layers.0.block.norm1.weight" in key: di = state_dict[key].shape[0]
                elif "core.audio_stack.layers.0.block.norm1.weight" in key: da = state_dict[key].shape[0]
                    
            return max(len(t_layers), default_text), max(len(i_layers), default_img), max(len(a_layers), default_audio), dt, di, da
        except Exception as e:
            print(f"[TopologyLoader] Failed to read checkpoint: {e}")
            return default_text, default_img, default_audio, def_dt, def_di, def_da

    def __init__(self, vocab_size: int = 50257, device: str = "cpu",
                 d_model_text: int = 128, d_model_img: int = 128, d_model_audio: int = 64,
                 layers_text: int = 2, layers_img: int = 1, layers_audio: int = 1):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        
        # 1. Core Physics Engine
        self.core = PCEIModel(
            vocab_size=vocab_size,
            d_model_text=d_model_text,
            d_model_img=d_model_img,
            d_model_audio=d_model_audio,
            layers_text=layers_text,
            layers_img=layers_img,
            layers_audio=layers_audio
        ).to(self.device)
        
        # 2. Consolidation & Subconscious
        self.ambient_field = AmbientField(self.core)
        
        # 3. Dynamic Topology
        self.neurogenesis = NeurogenesisController(self.core)
        
        # 4. Autonomy & Routing
        self.output_router = OutputRouter()
        self.response_gate = ResponseGate(d_model=d_model_text, settlement_threshold=0.05)
        
    def set_mode(self, mode: str):
        """Toggle between ACTIVE (chatting) and WITNESS (just observing)."""
        self.output_router.set_mode(mode)

    def adjust_plasticity(self, optimizer: torch.optim.Optimizer):
        """
        Calculates the effective learning rate for each PlasticLayer 
        based on its consolidation score and updates the optimizer.
        """
        from src.modules.plasticity import PlasticLayer
        for _, module in self.named_modules():
            if isinstance(module, PlasticLayer):
                module.adjust_learning_rates(optimizer)

    def forward(self, 
                text_tokens: Optional[torch.Tensor] = None,
                image_tensor: Optional[torch.Tensor] = None,
                audio_tensor: Optional[torch.Tensor] = None,
                text_latent: Optional[torch.Tensor] = None,
                image_latent: Optional[torch.Tensor] = None,
                audio_latent: Optional[torch.Tensor] = None,
                max_thinking_steps: int = 10) -> Optional[torch.Tensor]:
        """
        Executes a full forward pass through the PCEI system.
        Returns text logits if in ACTIVE mode and thought is settled.
        Returns None if in WITNESS mode.
        """
        
        # In a full run, we would loop over diffusion timesteps and check the ResponseGate.
        # For simplicity in this integration wrapper, we just run the fixed diffusion loop
        # defined in self.core(), and then apply the ResponseGate to the final latent.
        
        # Make sure inputs are on the right device
        if text_tokens is not None: text_tokens = text_tokens.to(self.device)
        if image_tensor is not None: image_tensor = image_tensor.to(self.device)
        if audio_tensor is not None: audio_tensor = audio_tensor.to(self.device)
        if text_latent is not None: text_latent = text_latent.to(self.device)
        if image_latent is not None: image_latent = image_latent.to(self.device)
        if audio_latent is not None: audio_latent = audio_latent.to(self.device)

        # Compute mean confidence across all PlasticLayers for dream_strength annealing
        from src.modules.plasticity import PlasticLayer
        confidence_vals = []
        for _, module in self.core.named_modules():
            if isinstance(module, PlasticLayer):
                val = module.slow_field_confidence.item()
                if torch.isnan(torch.tensor(val)): val = 0.0
                confidence_vals.append(val)
        mean_conf = sum(confidence_vals) / max(len(confidence_vals), 1)
        
        text_logits, t_latent, i_latent, a_latent, img_recon, audio_recon = self.core(
            text_input=text_tokens,
            img_input=image_tensor,
            audio_input=audio_tensor,
            text_latent=text_latent,
            img_latent=image_latent,
            audio_latent=audio_latent,
            diffusion_timesteps=max_thinking_steps,
            slow_state=self.ambient_field.slow_state,
            mean_confidence=mean_conf
        )
        
        # 2. Autonomy Check
        # Did the TextStack latent representation stabilize enough to warrant a response?
        thought_settled = True
        if t_latent is not None:
            thought_settled = self.response_gate.check_settlement(t_latent)
            
        # 3. Ambient Consolidation Tick
        # We assume the backward pass happens outside this forward func during open learning.
        # We manually tick the subconscious field just to simulate the passage of time.
        self.ambient_field.tick_consolidation()
        
        # 4. Structural Evaluation
        # Check if the gradients from the recent learning demand a new layer
        structural_actions = self.neurogenesis.evaluate_topology()
        if structural_actions:
            self.neurogenesis.apply_structural_changes(structural_actions)
            
        # 5. Output Routing
        # The model only outputs text, and only if it's in ACTIVE mode AND thought settled.
        if not thought_settled:
            return None  # Thought still forming — withhold output
        return self.output_router.route_output(text_logits)
