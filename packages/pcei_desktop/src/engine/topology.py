import torch
import torch.nn as nn
from typing import Dict, Tuple


from src.modules.plasticity import PlasticLayer

class TopologyShifter:
    """
    Manages the physical arrangement of PlasticLayers.
    Reads the active bonds from the EntanglementBroker and applies
    Pseudo-Gravitational forces to pull correlated layers closer in the X/Y plane.
    """
    def __init__(self, model: nn.Module, dt: float = 0.01, z_penalty_scalar: float = 0.1):
        self.model = model
        self.dt = dt # Simulation timestep size for movement
        self.z_penalty_scalar = z_penalty_scalar # How much Z-distance reduces the pull
        
        # Flattened registry of all PlasticLayers in the engine
        self.layers: Dict[str, PlasticLayer] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, PlasticLayer):
                # Ensure the layer_id is consistent
                self.layers[module.layer_id] = module

    def refresh_registry(self):
        """
        Re-scans the model for all PlasticLayers and updates the registry.
        Must be called after neurogenesis spawns new layers so the shifter
        doesn't miss newly grown topology.
        """
        from src.modules.plasticity import PlasticLayer
        for name, module in self.model.named_modules():
            if isinstance(module, PlasticLayer):
                if module.layer_id not in self.layers:
                    self.layers[module.layer_id] = module

    def apply_forces(self, active_bonds: Dict[Tuple[str, str], float]):
        """
        Calculates and applies the migration vectors for all layers based on bonds.
        """
        # Dictionary to accumulate movement forces: {layer_id: [delta_x, delta_y]}
        forces = {layer_id: [0.0, 0.0] for layer_id in self.layers.keys()}
        
        # Calculate attraction based on entanglement bonds
        for (id_a, id_b), strength in active_bonds.items():
            if id_a not in self.layers or id_b not in self.layers:
                continue
                
            layer_a = self.layers[id_a]
            layer_b = self.layers[id_b]
            
            pos_a = layer_a.get_position()
            pos_b = layer_b.get_position()
            
            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]
            dz = pos_b[2] - pos_a[2]
            
            # Simple inverse-square attenuation based on Z depth difference
            # Layers far apart in depth (e.g. Text vs Audio) pull each other less
            z_distance = abs(dz)
            attenuation = 1.0 / (1.0 + self.z_penalty_scalar * z_distance)
            
            # Hooke's Law style spring force: F = -kx (but applied symmetrically)
            # strength is our 'k'.
            force_x = dx * strength * attenuation * self.dt
            force_y = dy * strength * attenuation * self.dt
            
            # SYMMETRY BREAKER: If layers are perfectly stacked (dx=0, dy=0), 
            # they will never move because force is 0. Add tiny random jitter.
            if abs(dx) < 1e-6 and abs(dy) < 1e-6 and strength > 0:
                jitter = (torch.rand(2) - 0.5) * 1e-4
                force_x += jitter[0].item()
                force_y += jitter[1].item()
            
            # A pulls towards B
            forces[id_a][0] += force_x
            forces[id_a][1] += force_y
            
            # B pulls towards A (Equal and opposite)
            forces[id_b][0] -= force_x
            forces[id_b][1] -= force_y

        # Apply movements
        for layer_id, (fx, fy) in forces.items():
            self.layers[layer_id].update_position(fx, fy)

    def calculate_proximity_reward(self, id_a: str, id_b: str) -> float:
        """
        Returns a scalar reward/bonus based on how close two layers are in the X/Y plane.
        This provides the metabolic incentive: parameters communicating over shorter
        topological distances consume less virtual energy. 
        In PCEI, this reward can be added to the loss function or used by Neurogenesis.
        """
        if id_a not in self.layers or id_b not in self.layers:
            return 0.0
            
        pos_a = self.layers[id_a].get_position()
        pos_b = self.layers[id_b].get_position()
        
        # 2D Euclidean distance
        dist = ((pos_b[0] - pos_a[0])**2 + (pos_b[1] - pos_a[1])**2)**0.5
        
        # Reward is higher when distance is lower.
        # e.g., max reward of 1.0 at 0 distance, decaying towards 0.
        reward = 1.0 / (1.0 + dist)
        return reward
