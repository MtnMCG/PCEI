import torch
import pytest


from src.engine.autonomy import ResponseGate, OutputRouter

def test_response_gate_settlement():
    # Threshold is 0.05 max change
    gate = ResponseGate(d_model=64, settlement_threshold=0.05)
    
    # Tick 1: Initial state, gate is closed (gathering velocity base)
    x1 = torch.zeros(2, 64)
    assert gate.check_settlement(x1) is False, "Gate should be closed on first tick"
    
    # Tick 2: Huge change (unstable thought)
    x2 = torch.ones(2, 64) * 0.5 # Change is 0.5 > 0.05
    assert gate.check_settlement(x2) is False, "Gate should be closed when change is large"
    
    # Tick 3: Small change (settling)
    x3 = x2 + 0.02 # Change is 0.02 < 0.05
    assert gate.check_settlement(x3) is True, "Gate should open when representation settles"
    
    gate.reset()

def test_output_router_modes():
    router = OutputRouter()
    
    logits = torch.randn(2, 10, 100)
    
    # Active mode
    assert router.route_output(logits) is not None, "Router blocked output in ACTIVE mode"
    
    # Witness mode
    router.set_mode("WITNESS")
    assert router.route_output(logits) is None, "Router leaked output in WITNESS mode"
