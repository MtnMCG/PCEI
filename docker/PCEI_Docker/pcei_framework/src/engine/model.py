import torch
import torch.nn as nn
from typing import Dict, Optional


from src.core.stack import TextStack, ImageStack, AudioStack
from src.core.boundary import BoundaryLayer
from src.engine.seeder import DreamSeeder
from src.engine.physics import EntanglementBroker, ResonanceDetector
from src.engine.topology import TopologyShifter

# Default sequence lengths for each modality (used by BoundaryLayer temporal adapters)
TEXT_SEQ_LEN = 512
IMG_SEQ_LEN = 4   # 32x32 image / 16x16 patches = 4 patches
AUDIO_SEQ_LEN = 100


class PCEIModel(nn.Module):
    """
    The top-level integration hub connecting the Text, Image, and Audio stacks.
    
    Now with all three spec gaps closed:
    1. DreamSeeder primes noise before every diffusion timestep
    2. Stacks run in parallel via ThreadPoolExecutor
    3. (Bond dissolution handled in EntanglementBroker)
    """
    def __init__(self, vocab_size: int = 50257,
                 text_seq_len: int = TEXT_SEQ_LEN,
                 img_seq_len: int = IMG_SEQ_LEN,
                 audio_seq_len: int = AUDIO_SEQ_LEN,
                 d_model_text: int = 128,
                 d_model_img: int = 128,
                 d_model_audio: int = 64,
                 layers_text: int = 2,
                 layers_img: int = 1,
                 layers_audio: int = 1):
        super().__init__()
        
        self.text_seq_len = text_seq_len
        self.img_seq_len = img_seq_len
        self.audio_seq_len = audio_seq_len
        
        # Instantiate stacks directly based on spec
        # COLLAPSED Z-AXIS: Move stacks closer for resonance efficiency
        self.text_stack = TextStack(vocab_size=vocab_size, d_model=d_model_text, num_layers=layers_text, start_z=0.0)
        self.img_stack = ImageStack(d_model=d_model_img, num_layers=layers_img, start_z=2.0)
        self.audio_stack = AudioStack(d_model=d_model_audio, num_layers=layers_audio, start_z=4.0)
        
        # DreamSeeder — primes noise at each diffusion timestep
        self.dream_seeder = DreamSeeder(
            d_model_text=d_model_text, d_model_img=d_model_img, d_model_audio=d_model_audio
        )
        
        # Full-sequence Boundary Layers with temporal adapters
        self.boundary_t_i = BoundaryLayer(
            d_model_upper=d_model_text, d_model_lower=d_model_img,
            seq_len_upper=text_seq_len, seq_len_lower=img_seq_len
        )
        self.boundary_i_a = BoundaryLayer(
            d_model_upper=d_model_img, d_model_lower=d_model_audio,
            seq_len_upper=img_seq_len, seq_len_lower=audio_seq_len
        )
        
        # NOTE: Stacks execute sequentially. ThreadPoolExecutor was removed because
        # Python's GIL and MPS single-stream dispatch mean threads provide zero
        # speedup while introducing shared-memory race conditions.
        
        # ── Milestone 7: Physics Engine Integration ──
        self.broker = EntanglementBroker(correlation_threshold=0.85)
        self.shifter = TopologyShifter(self)
        self.detectors = {}
        for name, module in self.named_modules():
            from src.modules.plasticity import PlasticLayer
            if isinstance(module, PlasticLayer):
                self.detectors[module.layer_id] = ResonanceDetector(module.layer_id, module.d_model)

        # Scale factor for dream noise injection (small to not destabilize early training)
        self.dream_injection_scale = 0.1

    def _tick_text_stack(self, t_latent: torch.Tensor, t_tensor: torch.Tensor,
                         dream_seed: Optional[torch.Tensor],
                         spike: Optional[torch.Tensor] = None,
                         detectors: Dict[str, ResonanceDetector] = None) -> torch.Tensor:
        """Process one diffusion step for the text stack."""
        if dream_seed is not None:
            # Inject dream-primed noise: match sequence length
            if dream_seed.size(1) != t_latent.size(1):
                dream_seed = torch.nn.functional.interpolate(
                    dream_seed.transpose(1, 2), size=t_latent.size(1), mode='linear', align_corners=False
                ).transpose(1, 2)
            t_latent = t_latent + dream_seed * self.dream_injection_scale
            
        # LIVE PHYSICS: Inject sync spikes from the EntanglementBroker
        if spike is not None:
            # Match sequence length if necessary (spikes are pooled per-layer energy)
            if spike.dim() == 2: # (B, d_model)
                spike = spike.unsqueeze(1).expand_as(t_latent)
            t_latent = t_latent + spike

        return self.text_stack(t_latent, t_tensor, detectors=detectors)
    
    def _tick_img_stack(self, i_latent: torch.Tensor, t_tensor: torch.Tensor,
                        dream_seed: Optional[torch.Tensor],
                        spike: Optional[torch.Tensor] = None,
                        detectors: Dict[str, ResonanceDetector] = None) -> torch.Tensor:
        """Process one diffusion step for the image stack."""
        if dream_seed is not None:
            if dream_seed.size(1) != i_latent.size(1):
                dream_seed = torch.nn.functional.interpolate(
                    dream_seed.transpose(1, 2), size=i_latent.size(1), mode='linear', align_corners=False
                ).transpose(1, 2)
            i_latent = i_latent + dream_seed * self.dream_injection_scale
            
        if spike is not None:
            if spike.dim() == 2:
                spike = spike.unsqueeze(1).expand_as(i_latent)
            i_latent = i_latent + spike

        return self.img_stack(i_latent, t_tensor, detectors=detectors)
    
    def _tick_audio_stack(self, a_latent: torch.Tensor, t_tensor: torch.Tensor,
                          dream_seed: Optional[torch.Tensor],
                          spike: Optional[torch.Tensor] = None,
                          detectors: Dict[str, ResonanceDetector] = None) -> torch.Tensor:
        """Process one diffusion step for the audio stack."""
        if dream_seed is not None:
            if dream_seed.size(1) != a_latent.size(1):
                dream_seed = torch.nn.functional.interpolate(
                    dream_seed.transpose(1, 2), size=a_latent.size(1), mode='linear', align_corners=False
                ).transpose(1, 2)
            a_latent = a_latent + dream_seed * self.dream_injection_scale

        if spike is not None:
            if spike.dim() == 2:
                spike = spike.unsqueeze(1).expand_as(a_latent)
            a_latent = a_latent + spike

        return self.audio_stack(a_latent, t_tensor, detectors=detectors)

    def forward(self, 
                text_input: Optional[torch.Tensor] = None,
                img_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None,
                text_latent: Optional[torch.Tensor] = None,
                img_latent: Optional[torch.Tensor] = None,
                audio_latent: Optional[torch.Tensor] = None,
                diffusion_timesteps: int = 10,
                slow_state: Optional[dict] = None,
                mean_confidence: float = 0.0,
                spikes: Optional[Dict[str, torch.Tensor]] = None,
                override_timestep: Optional[int] = None):
        """
        Runs the full multimodal forward pass over T diffusion timesteps.
        
        NEW: DreamSeeder primes noise at each timestep.
        NEW: Stacks execute sequentially for deterministic training on MPS.
        NEW: override_timestep — when set, the sinusoidal embedding uses this
             value instead of the loop counter, so training at sampled noise
             levels gets the correct timestep conditioning.
        """
        device = next(self.parameters()).device
        t_latent, i_latent, a_latent = None, None, None
        
        # Step 1: Ingestion — embed raw inputs or use provided latents
        if text_latent is not None:
            t_latent = text_latent
        elif text_input is not None:
            t_latent = self.text_stack.integrate_input(text_input)
            
        if img_latent is not None:
            i_latent = img_latent
        elif img_input is not None:
            batch = img_input.size(0)
            img_input = img_input.view(batch, -1, self.img_stack.patch_dim)
            i_latent = self.img_stack.integrate_input(img_input)
            
        if audio_latent is not None:
            a_latent = audio_latent
        elif audio_input is not None:
            batch = audio_input.size(0)
            audio_input = audio_input.view(batch, -1, self.audio_stack.mel_bins)
            a_latent = self.audio_stack.integrate_input(audio_input)
        
        # Determine batch size from whatever input we got
        batch_size = 1
        if t_latent is not None: batch_size = t_latent.size(0)
        elif i_latent is not None: batch_size = i_latent.size(0)
        elif a_latent is not None: batch_size = a_latent.size(0)
        
        # Build intent vector for DreamSeeder (check for mental anchor first)
        intent_vector = None
        if hasattr(self.dream_seeder, 'active_intent') and self.dream_seeder.active_intent is not None:
            intent_vector = self.dream_seeder.active_intent
        elif t_latent is not None:
            intent_vector = t_latent.mean(dim=1)  # (B, d_text)
        else:
            intent_vector = torch.zeros(batch_size, self.text_stack.d_model, device=device)
            
        # Physics Simulation - Iterative Denoising
        current_spikes = None
        for t in reversed(range(diffusion_timesteps)):
            # Use override if provided (for diffusion training at specific noise levels)
            effective_t = override_timestep if override_timestep is not None else t
            t_tensor = torch.full((1,), effective_t, dtype=torch.float32, device=device)
            
            # ── LIVE PHYSICS: Inject sync spikes from previous step ──
            # (In Step 1, spikes is None)
            
            # ── Step 2: DreamSeeder primes the noise ──
            seq_lens = {}
            if t_latent is not None: seq_lens['text'] = t_latent.size(1)
            if i_latent is not None: seq_lens['img'] = i_latent.size(1)
            if a_latent is not None: seq_lens['audio'] = a_latent.size(1)
            
            dream_seeds = self.dream_seeder(
                intent_vector=intent_vector,
                batch_size=batch_size,
                seq_lens=seq_lens,
                slow_state=slow_state,
                mean_confidence=mean_confidence
            )
            
            # ── Step 3: Clear boundary buffers ──
            self.boundary_t_i.clear_buffers()
            self.boundary_i_a.clear_buffers()
            
            # ── Step 3: Sequential stack processing ──
            # Each stack runs its full diffusion step independently, then 
            # results are collected for physics and boundary exchange.
            t_latent_next = t_latent
            i_latent_next = i_latent
            a_latent_next = a_latent
            
            if t_latent is not None:
                t_latent_next = self._tick_text_stack(
                    t_latent, t_tensor,
                    dream_seeds.get('text'),
                    current_spikes.get('TEXT_L_GLOBAL') if current_spikes else None,
                    self.detectors
                )
                # Circuit breaker: clamp NaN/Inf
                if torch.isnan(t_latent_next).any() or torch.isinf(t_latent_next).any():
                    print(f"[CIRCUIT BREAKER] NaN/Inf in TextStack at timestep {t}. Clamping.")
                    t_latent_next = torch.nan_to_num(t_latent_next, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            if i_latent is not None:
                i_latent_next = self._tick_img_stack(
                    i_latent, t_tensor,
                    dream_seeds.get('img'),
                    current_spikes.get('IMAGE_L_GLOBAL') if current_spikes else None,
                    self.detectors
                )
                if torch.isnan(i_latent_next).any() or torch.isinf(i_latent_next).any():
                    print(f"[CIRCUIT BREAKER] NaN/Inf in ImageStack at timestep {t}. Clamping.")
                    i_latent_next = torch.nan_to_num(i_latent_next, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            if a_latent is not None:
                a_latent_next = self._tick_audio_stack(
                    a_latent, t_tensor,
                    dream_seeds.get('audio'),
                    current_spikes.get('AUDIO_L_GLOBAL') if current_spikes else None,
                    self.detectors
                )
                if torch.isnan(a_latent_next).any() or torch.isinf(a_latent_next).any():
                    print(f"[CIRCUIT BREAKER] NaN/Inf in AudioStack at timestep {t}. Clamping.")
                    a_latent_next = torch.nan_to_num(a_latent_next, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Measure Entanglement and Generate Spikes for NEXT step
            bonds = self.broker.measure_entanglement(self.detectors)
            raw_spikes = self.broker.generate_sync_spikes(self.detectors, self.shifter)
            
            # Map raw spikes to global stack ports for the next tick
            # (This is where the "Physically Constrained" intelligence emerges)
            current_spikes = {}
            if raw_spikes:
                for lid, s in raw_spikes.items():
                    if "TEXT" in lid: current_spikes["TEXT_L_GLOBAL"] = s
                    elif "IMAGE" in lid: current_spikes["IMAGE_L_GLOBAL"] = s
                    elif "AUDIO" in lid: current_spikes["AUDIO_L_GLOBAL"] = s
            
            # ── Step 4: Boundary Layer Exchange ──
            if t_latent is not None:
                self.boundary_t_i.send_down(t_latent_next, target_seq_len=i_latent.size(1) if i_latent is not None else None)
            if i_latent is not None:
                self.boundary_t_i.send_up(i_latent_next, target_seq_len=t_latent.size(1) if t_latent is not None else None)
                self.boundary_i_a.send_down(i_latent_next, target_seq_len=a_latent.size(1) if a_latent is not None else None)
            if a_latent is not None:
                self.boundary_i_a.send_up(a_latent_next, target_seq_len=i_latent.size(1) if i_latent is not None else None)
            
            # Cross-modal signal integration
            if t_latent is not None:
                img_signal = self.boundary_t_i.receive_up()
                if img_signal is not None:
                    t_latent_next = t_latent_next + img_signal
                t_latent = t_latent_next
            
            if i_latent is not None:
                text_signal = self.boundary_t_i.receive_down()
                aud_signal = self.boundary_i_a.receive_up()
                if text_signal is not None: i_latent_next = i_latent_next + text_signal
                if aud_signal is not None: i_latent_next = i_latent_next + aud_signal
                i_latent = i_latent_next
            
            if a_latent is not None:
                img_signal_down = self.boundary_i_a.receive_down()
                if img_signal_down is not None:
                    a_latent_next = a_latent_next + img_signal_down
                a_latent = a_latent_next

        # Output
        text_logits = None
        if t_latent is not None:
            text_logits = self.text_stack.decode_output(t_latent)
        
        # Reconstruct image/audio for multimodal supervision (Phase 3)
        img_recon = None
        if i_latent is not None and hasattr(self.img_stack, 'decode_output'):
            img_recon = self.img_stack.decode_output(i_latent)
        
        audio_recon = None
        if a_latent is not None and hasattr(self.audio_stack, 'decode_output'):
            audio_recon = self.audio_stack.decode_output(a_latent)
            
        return text_logits, t_latent, i_latent, a_latent, img_recon, audio_recon
