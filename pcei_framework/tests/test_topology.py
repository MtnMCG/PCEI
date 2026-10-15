import torch
import pytest


from src.modules.plasticity import PlasticLayer
from src.engine.topology import TopologyShifter

class DummyPCEIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Place two layers down in space at Z=0 and Z=10
        self.layer_a = PlasticLayer(layer_id="A", d_model=64, initial_pos=(0.0, 0.0, 0.0))
        self.layer_b = PlasticLayer(layer_id="B", d_model=64, initial_pos=(5.0, 5.0, 10.0))

def test_topology_migration_and_reward():
    model = DummyPCEIModel()
    
    # Init Shifter
    shifter = TopologyShifter(model=model, dt=1.0, z_penalty_scalar=0.1)
    
    # Store initial positions
    pos_a_init = model.layer_a.get_position()
    pos_b_init = model.layer_b.get_position()
    
    # The layers start far apart
    assert pos_a_init == (0.0, 0.0, 0.0)
    assert pos_b_init == (5.0, 5.0, 10.0)
    
    # Calculate proximity reward before
    reward_before = shifter.calculate_proximity_reward("A", "B")
    
    # Apply a strong entanglement bond between A and B
    active_bonds = {("A", "B"): 0.9} 
    
    # Tick physics engine
    shifter.apply_forces(active_bonds)
    
    pos_a_new = model.layer_a.get_position()
    pos_b_new = model.layer_b.get_position()
    
    # Calculate proximity reward after
    reward_after = shifter.calculate_proximity_reward("A", "B")
    
    # A should have moved towards B (Positive X/Y)
    assert pos_a_new[0] > pos_a_init[0], "Layer A did not move towards +X"
    assert pos_a_new[1] > pos_a_init[1], "Layer A did not move towards +Y"
    
    # B should have moved towards A (Negative X/Y)
    assert pos_b_new[0] < pos_b_init[0], "Layer B did not move towards -X"
    assert pos_b_new[1] < pos_b_init[1], "Layer B did not move towards -Y"
    
    # Z should be entirely unchanged
    assert pos_a_new[2] == 0.0
    assert pos_b_new[2] == 10.0
    
    # Physical proximity reward (bonus) should increase as they get closer
    assert reward_after > reward_before, "Proximity reward did not increase after migration"
