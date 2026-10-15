import torch
import pytest

from src.engine.physics import ResonanceDetector, EntanglementBroker

def test_resonance_detector_buffer():
    detector = ResonanceDetector(layer_id="l1", d_model=64, buffer_size=3)
    
    x1 = torch.randn(2, 5, 64)
    x2 = torch.randn(2, 5, 64)
    x3 = torch.randn(2, 5, 64)
    x4 = torch.randn(2, 5, 64)
    
    detector.register_activation(x1)
    detector.register_activation(x2)
    detector.register_activation(x3)
    
    sig = detector.get_temporal_signature()
    assert sig is not None
    # register_activation pools to (d_model,), stack on dim=0 → (time, d_model)
    assert sig.shape == (3, 64), f"Expected shape (3, 64), got {sig.shape}"
    
    detector.register_activation(x4)
    sig2 = detector.get_temporal_signature()
    # buffer_size=3, so 4th activation pops the oldest → still 3 in buffer
    assert sig2.shape == (3, 64), f"Expected shape (3, 64), got {sig2.shape}"

def test_entanglement_broker_correlation():
    broker = EntanglementBroker(correlation_threshold=0.85)
    
    det_a = ResonanceDetector(layer_id="stack_A_L2", d_model=128, buffer_size=10)
    det_b = ResonanceDetector(layer_id="stack_B_L1", d_model=256, buffer_size=10)
    det_c = ResonanceDetector(layer_id="stack_C_L1", d_model=64, buffer_size=10)
    
    batch = 4
    for t in range(10):
        pulse = 10.0 if t == 5 else 0.0
        
        x_a = torch.randn(batch, 5, 128) + pulse
        x_b = torch.randn(batch, 5, 256) + pulse
        x_c = torch.randn(batch, 5, 64)
        
        det_a.register_activation(x_a)
        det_b.register_activation(x_b)
        det_c.register_activation(x_c)
        
    detectors = {"stack_A": det_a, "stack_B": det_b, "stack_C": det_c}
    bonds = broker.measure_entanglement(detectors)
    
    bond_key = ("stack_A", "stack_B")
    assert bond_key in bonds or ("stack_B", "stack_A") in bonds

def test_sync_spike_injection():
    """Verify that entangled layers receive non-zero synchronizing spikes."""
    broker = EntanglementBroker(correlation_threshold=0.85, spike_scale=0.1)
    
    det_a = ResonanceDetector(layer_id="A", d_model=64, buffer_size=10)
    det_b = ResonanceDetector(layer_id="B", d_model=128, buffer_size=10)
    
    batch = 2
    for t in range(10):
        pulse = 5.0 if t == 5 else 0.0
        det_a.register_activation(torch.randn(batch, 3, 64) + pulse)
        det_b.register_activation(torch.randn(batch, 3, 128) + pulse)
    
    detectors = {"A": det_a, "B": det_b}
    bonds = broker.measure_entanglement(detectors)
    
    # Generate spikes
    spikes = broker.generate_sync_spikes(detectors)
    
    if ("A", "B") in bonds or ("B", "A") in bonds:
        # At least one spike should be non-zero
        has_spike = False
        for layer_id, spike in spikes.items():
            if spike.abs().sum() > 0:
                has_spike = True
        assert has_spike, "Spike injection produced zero spikes despite active bond"
