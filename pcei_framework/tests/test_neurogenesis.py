import torch
import pytest


from src.modules.plasticity import PlasticLayer
from src.engine.neurogenesis import NeurogenesisController

class DummyPCEIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = PlasticLayer(layer_id="A", d_model=64, initial_pos=(0.0, 0.0, 0.0))
        self.layer_b = PlasticLayer(layer_id="B", d_model=64, initial_pos=(0.0, 0.0, 1.0))
        self.layer_c = PlasticLayer(layer_id="C", d_model=64, initial_pos=(0.0, 0.0, 2.0))

def test_neurogenesis_spawn_action():
    model = DummyPCEIModel()
    controller = NeurogenesisController(model=model, spawn_threshold=1.0, prune_threshold=0.01)
    controller.enabled = True
    controller.warmup_steps = 0
    
    # Set high confidence and high weight norm to satisfy SPAWN preconditions
    model.layer_a.slow_field_confidence.copy_(torch.tensor([0.99]))
    with torch.no_grad():
        model.layer_a.block.mamba.in_proj.weight.fill_(5.0)
    
    x = torch.randn(2, 5, 64, requires_grad=True)
    out = model.layer_a(x, torch.zeros(2))
    
    loss = (out.sum() * 50.0).pow(2)
    loss.backward()
    
    controller.stress_history["B"] = 0.0
    controller.stress_history["C"] = 0.0
    for _ in range(5):
        controller.monitor_gradients()
        
    actions = controller.evaluate_topology()
    
    assert "A" in actions, "Layer A was not evaluated for action"
    assert actions["A"] == "SPAWN", f"Expected SPAWN, got {actions['A']}"

def test_neurogenesis_prune_action():
    model = DummyPCEIModel()
    controller = NeurogenesisController(model=model, spawn_threshold=5.0, prune_threshold=0.1)
    controller.enabled = True
    controller.warmup_steps = 0
    
    model.layer_a.slow_field_confidence.copy_(torch.tensor([0.99]))
    model.zero_grad()
    
    for _ in range(5):
        controller.monitor_gradients()
        
    actions = controller.evaluate_topology()
    
    assert "A" in actions, "Layer A was not evaluated for action"
    assert actions["A"] == "PRUNE", f"Expected PRUNE, got {actions['A']}"

