import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path
import math

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
    def build_physics_optimizer(cls, nexus, base_lr: float, weight_decay: float = 0.01):
        """
        Builds an AdamW optimizer where every PlasticLayer gets its own parameter group.
        This is absolutely required for the Ambient Field (adjust_plasticity) to dynamically
        throttle learning rates per layer to prevent catastrophic forgetting.
        """
        from src.modules.plasticity import PlasticLayer
        import torch
        
        param_groups = []
        plastic_param_ids = set()
        
        # 1. Add each PlasticLayer as its own parameter group
        for _, module in nexus.named_modules():
            if isinstance(module, PlasticLayer):
                layer_params = [p for p in module.parameters() if p.requires_grad]
                if layer_params:
                    decay = [p for p in layer_params if p.dim() >= 2]
                    no_decay = [p for p in layer_params if p.dim() < 2]
                    if decay:
                        param_groups.append({'params': decay, 'lr': base_lr, 'weight_decay': weight_decay})
                    if no_decay:
                        param_groups.append({'params': no_decay, 'lr': base_lr, 'weight_decay': 0.0})
                    for p in layer_params:
                        plastic_param_ids.add(id(p))
                        
        # 2. Group all other parameters (embeddings, heads, boundary layers) together
        other_params = [p for p in nexus.parameters() if id(p) not in plastic_param_ids and p.requires_grad]
        if other_params:
            decay = [p for p in other_params if p.dim() >= 2]
            no_decay = [p for p in other_params if p.dim() < 2]
            if decay:
                param_groups.append({'params': decay, 'lr': base_lr, 'weight_decay': weight_decay})
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': base_lr, 'weight_decay': 0.0})
            
        return torch.optim.AdamW(param_groups, foreach=False)

    @classmethod
    def load_topology_from_checkpoint(cls, ckpt_path: Path, default_text=12, default_img=4, default_audio=4, def_dt=768, def_di=512, def_da=256):
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

    @staticmethod
    def save_safe_checkpoint(nexus_instance, state_dict_to_save: dict, save_path: Path, create_backup: bool = False) -> bool:
        """
        The Golden Rule Checkpoint Guard.
        Rapidly scans the model for NaNs before saving. If corrupted, aborts the save.
        If healthy, saves to a temporary file atomically, replaces the main file, 
        and optionally creates a known-good backup.
        """
        # 1. NaN Validation
        for name, p in nexus_instance.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"\n[CRITICAL ERROR] NaN/Inf detected in parameter: {name}")
                print("[CRITICAL ERROR] ABORTING SAVE to protect checkpoint from corruption.")
                return False
                
        # Block Ctrl+C during the atomic save so the user doesn't accidentally corrupt it
        import signal
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Inject AmbientField consciousness
        if hasattr(nexus_instance, 'ambient_field'):
            state_dict_to_save['ambient_slow_state'] = nexus_instance.ambient_field.slow_state
            state_dict_to_save['ambient_fast_state'] = nexus_instance.ambient_field.fast_state
            state_dict_to_save['ambient_layer_emas'] = nexus_instance.ambient_field.layer_emas

        # Inject Entanglement Bond state so the dashboard can render the topology
        if hasattr(nexus_instance, 'core') and hasattr(nexus_instance.core, 'broker'):
            broker = nexus_instance.core.broker
            if hasattr(broker, 'active_bonds'):
                state_dict_to_save['active_bonds'] = dict(broker.active_bonds)
            if hasattr(broker, 'bonds_formed_total'):
                state_dict_to_save['bonds_formed_total'] = broker.bonds_formed_total
            if hasattr(broker, 'bonds_dissolved_total'):
                state_dict_to_save['bonds_dissolved_total'] = broker.bonds_dissolved_total
            
        try:
            # 2. Atomic Save (Write to .tmp first)
            tmp_path = str(save_path) + '.tmp'
            torch.save(state_dict_to_save, tmp_path)
            
            # 3. Intelligent Backup
            if create_backup and save_path.exists():
                backup_path = save_path.with_name(f"{save_path.stem}_LAST_KNOWN_GOOD.pt")
                import shutil
                shutil.copy2(save_path, backup_path)
                
            # 4. Atomic Replace
            import os
            os.replace(tmp_path, str(save_path))
        finally:
            # Restore Ctrl+C
            signal.signal(signal.SIGINT, original_handler)
            
        return True

    def __init__(self, config=None, device: str = "cpu"):
        super().__init__()
        
        if config is None:
            from src.config import PCEIConfig
            self.config = PCEIConfig()
        else:
            self.config = config
            
        self.device = torch.device(device)
        self.vocab_size = self.config.vocab_size
        
        # Determine best dtype
        if self.config.system_dtype == "float16":
            self.system_dtype = torch.float16
        elif self.config.system_dtype == "bfloat16" and hasattr(torch, 'bfloat16'):
            self.system_dtype = torch.bfloat16
        else:
            self.system_dtype = torch.float32
        
        # 1. Core Physics Engine
        self.core = PCEIModel(
            config=self.config
        ).to(self.device).to(self.system_dtype)
        
        # 2. Consolidation & Subconscious
        self.ambient_field = AmbientField(self.core)
        
        # 3. Dynamic Topology
        self.neurogenesis = NeurogenesisController(self.core)
        
        # 4. Autonomy & Routing
        self.output_router = OutputRouter()
        self.response_gate = ResponseGate(d_model=self.config.d_model_text, settlement_threshold=0.05)
        self._structure_changed = False
        
    def set_mode(self, mode: str):
        """Toggle between ACTIVE (chatting) and WITNESS (just observing)."""
        self.output_router.set_mode(mode)
        
    def load_ambient_state(self, ckpt_dict: dict):
        """Loads the persistent deep memory of the network if it exists."""
        if 'ambient_slow_state' in ckpt_dict and ckpt_dict['ambient_slow_state'] is not None:
            self.ambient_field.slow_state = ckpt_dict['ambient_slow_state'].to(self.device)
            if 'ambient_fast_state' in ckpt_dict and ckpt_dict['ambient_fast_state'] is not None:
                self.ambient_field.fast_state = ckpt_dict['ambient_fast_state'].to(self.device)
            if 'ambient_layer_emas' in ckpt_dict:
                self.ambient_field.layer_emas = ckpt_dict['ambient_layer_emas']
            print("[AmbientField] Persistent consciousness loaded from checkpoint.")
        else:
            print("[AmbientField] No persistent consciousness found. Starting fresh.")

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
                max_thinking_steps: int = 10,
                cache=None) -> Optional[torch.Tensor]:
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
                if math.isnan(val): val = 0.0
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
            mean_confidence=mean_conf,
            cache=cache
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
            structure_changed = self.neurogenesis.apply_structural_changes(structural_actions)
            if structure_changed:
                self._structure_changed = True
            
        # 5. Output Routing
        # The model only outputs text, and only if it's in ACTIVE mode AND thought settled.
        if not thought_settled:
            return None  # Thought still forming — withhold output
        return self.output_router.route_output(text_logits)
