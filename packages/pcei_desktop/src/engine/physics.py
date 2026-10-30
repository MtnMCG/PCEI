import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class ResonanceDetector(nn.Module):
    """
    Monitors a PlasticLayer's output over time (across diffusion steps or frames)
    to detect "spikes" — significant activation energy patterns.
    Maintains a rolling temporal buffer.
    """
    def __init__(self, layer_id: str, d_model: int, buffer_size: int = 10):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = d_model
        self.buffer_size = buffer_size
        
        self.activation_buffer: List[torch.Tensor] = []
        
    def register_activation(self, x: torch.Tensor):
        """
        x: (batch, seq, d_model) or (batch, d_model)
        Globally pooled to (d_model,) for robust temporal correlation regardless of batch size
        """
        if x.dim() == 3:
            x_pooled = x.mean(dim=(0, 1)).detach()  # (d_model,)
        else:
            x_pooled = x.mean(dim=0).detach()
        self.activation_buffer.append(x_pooled)
        
        if len(self.activation_buffer) > self.buffer_size:
            self.activation_buffer.pop(0)

    def get_temporal_signature(self) -> Optional[torch.Tensor]:
        """Returns (batch, time, d_model) or None if insufficient data."""
        if len(self.activation_buffer) < 2:
            return None
        return torch.stack(self.activation_buffer, dim=0)


class EntanglementBroker(nn.Module):
    """
    Physics Engine: Cross-Modal Sync via Quantum Brain Theory.
    
    Full bond lifecycle per spec:
    - Formation: correlation exceeds θ_form → bond created
    - Active: synchronizing spikes injected per forward pass
    - Decay: bond strength decays by decay_rate each tick  
    - Dissolution: bond dissolves when strength drops below θ_dissolve
    
    Bonds are dynamic and temporary — they must be continuously
    reinforced by real cross-correlation to survive.
    """
    def __init__(self, correlation_threshold: float = 0.80, reinforce_threshold: float = 0.65, spike_scale: float = 0.01,
                 decay_rate: float = 0.15, dissolve_threshold: float = 0.1):
        super().__init__()
        self.correlation_threshold = correlation_threshold  # minimum correlation to form a bond
        self.reinforce_threshold = reinforce_threshold      # minimum correlation to prevent decay
        self.dissolve_threshold = dissolve_threshold        # θ_dissolve
        self.decay_rate = decay_rate                        # Strength lost per tick
        
        # Learned spike scale
        self.spike_scale = nn.Parameter(torch.tensor(spike_scale))
        
        # Active bonds: {(layer_id_A, layer_id_B): bond_strength}
        self.active_bonds: Dict[Tuple[str, str], float] = {}
        
        # Bond metadata for logging
        self.bonds_formed_total = 0
        self.bonds_dissolved_total = 0

    def tick_decay(self):
        """
        Called once per training step. All bond strengths decay naturally.
        Bonds below θ_dissolve are dissolved — parameters do NOT un-migrate.
        """
        dissolved = []
        for bond_key, strength in self.active_bonds.items():
            self.active_bonds[bond_key] = strength - self.decay_rate
            if self.active_bonds[bond_key] < self.dissolve_threshold:
                dissolved.append(bond_key)
        
        for key in dissolved:
            del self.active_bonds[key]
            self.bonds_dissolved_total += 1

    def measure_entanglement(self, detectors: Dict[str, ResonanceDetector]) -> Dict[Tuple[str, str], float]:
        """
        Compares all registered detectors via Pearson correlation of relative temporal energy.
        High correlation → forms or reinforces a bond.
        Low correlation → bond decays naturally via tick_decay().
        """
        signatures = {}
        for layer_id, detector in detectors.items():
            sig = detector.get_temporal_signature()
            if sig is not None:
                signatures[layer_id] = sig
                
        if len(signatures) < 2:
            return self.active_bonds
            
        layer_ids = list(signatures.keys())
        
        # 1. Compute individual energies and find global minimum time window
        layer_energies = {}
        min_t = min([sig.shape[0] for sig in signatures.values()])
        
        if min_t < 2:
            return self.active_bonds
            
        for layer_id, sig in signatures.items():
            sig_matched = sig[-min_t:, :]
            energy = (sig_matched ** 2).mean(dim=-1)  # (time,)
            layer_energies[layer_id] = energy
            
        # 2. Compute global mean energy profile across the network
        all_energies = torch.stack(list(layer_energies.values())) # (num_layers, time)
        global_energy = all_energies.mean(dim=0) # (time,)
        
        # 3. Compute relative correlations (all pairs — same-stack AND cross-modal)
        for i in range(len(layer_ids)):
            for j in range(i + 1, len(layer_ids)):
                id_a = layer_ids[i]
                id_b = layer_ids[j]
                
                # Subtract global energy to isolate local feature syncing
                energy_a = layer_energies[id_a] - global_energy
                energy_b = layer_energies[id_b] - global_energy
                
                # Pearson correlation of relative temporal energy profiles
                energy_a = energy_a - energy_a.mean(dim=-1, keepdim=True)
                energy_b = energy_b - energy_b.mean(dim=-1, keepdim=True)
                
                std_a = energy_a.std(dim=-1, keepdim=True) + 1e-6
                std_b = energy_b.std(dim=-1, keepdim=True) + 1e-6
                
                energy_a = energy_a / std_a
                energy_b = energy_b / std_b
                
                corr = (energy_a * energy_b).mean(dim=-1)  # (scale)
                mean_corr = corr.mean().item()
                
                bond_key = (id_a, id_b)
                
                if bond_key not in self.active_bonds:
                    # Formation: only if correlation exceeds the high bar
                    if mean_corr > self.correlation_threshold:
                        self.bonds_formed_total += 1
                        self.active_bonds[bond_key] = 1.0  # Born at full strength
                else:
                    # Reinforcement: weighted by how far above threshold the correlation is
                    if mean_corr > self.reinforce_threshold:
                        # Scale boost by excess correlation: barely above = tiny boost
                        excess = (mean_corr - self.reinforce_threshold) / (1.0 - self.reinforce_threshold + 1e-6)
                        self.active_bonds[bond_key] = min(1.0, self.active_bonds[bond_key] + 0.02 * excess)
                    else:
                        # Active penalty: correlation dropped, accelerate decay
                        self.active_bonds[bond_key] -= self.decay_rate
        
        # Natural decay no longer called here to prevent over-decay during diffusion
        return self.active_bonds

    def generate_sync_spikes(self, detectors: Dict[str, ResonanceDetector], shifter: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        Quantum Brain Theory spike injection — Now with Physical Attenuation.
        
        For each active bond:
        - Spike magnitude is proportional to bond_strength * proximity_factor
        - proximity_factor = 1.0 / (1.0 + distance)
        - This provides the metabolic incentive to move closer.
        """
        spikes: Dict[str, torch.Tensor] = {}
        
        for (id_a, id_b), strength in self.active_bonds.items():
            if id_a not in detectors or id_b not in detectors:
                continue
            
            # Metabolic Pressure: Attenuate spikes by distance
            proximity_factor = 1.0
            if shifter is not None and hasattr(shifter, 'calculate_proximity_reward'):
                proximity_factor = shifter.calculate_proximity_reward(id_a, id_b)
            
            buf_a = detectors[id_a].activation_buffer
            buf_b = detectors[id_b].activation_buffer
            
            if len(buf_a) == 0 or len(buf_b) == 0:
                continue
                
            latest_a = buf_a[-1]  # (batch, d_model_a)
            latest_b = buf_b[-1]  # (batch, d_model_b)
            
            energy_b = latest_b.norm(dim=-1, keepdim=True)  # (batch, 1)
            energy_a = latest_a.norm(dim=-1, keepdim=True)
            
            spike_magnitude = strength * self.spike_scale * proximity_factor
            
            # Capped Spike Projection: Spikes are directions, not energy injectors.
            # We normalize the spike and scale it by a fraction (spike_magnitude) of the receiver's own norm.
            # Spike for A: direction of A, modulated by B's energy profile, but capped.
            spike_a = latest_a * (energy_b / (energy_b + 1e-6)) * spike_magnitude
            # Spike for B: direction of B, modulated by A's energy profile, but capped.
            spike_b = latest_b * (energy_a / (energy_a + 1e-6)) * spike_magnitude
            
            if id_a in spikes:
                spikes[id_a] = spikes[id_a] + spike_a
            else:
                spikes[id_a] = spike_a
                
            if id_b in spikes:
                spikes[id_b] = spikes[id_b] + spike_b
            else:
                spikes[id_b] = spike_b
                
        return spikes
