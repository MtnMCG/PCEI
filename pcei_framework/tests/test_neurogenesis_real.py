"""
Test: Real Neurogenesis — verify actual graph mutation
Tests that SPAWN physically adds layers and PRUNE physically removes them.
"""
import torch
import pytest

from src.engine.model import PCEIModel
from src.config import PCEIConfig
from src.engine.neurogenesis import NeurogenesisController
from src.modules.plasticity import PlasticLayer


def test_neurogenesis_spawn_actually_adds_layer():
    """Verify that SPAWN action physically inserts a new PlasticLayer into the stack."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=2, layers_img=1, layers_audio=1
    )
    model = PCEIModel(config=config)
    controller = NeurogenesisController(model, spawn_threshold=0.5)
    initial_count = len(model.text_stack.layers)
    
    first_layer = model.text_stack.layers[0]
    if hasattr(first_layer, 'module'):
        layer_id = first_layer.module.layer_id
    else:
        layer_id = first_layer.layer_id
    
    controller.apply_structural_changes({layer_id: "SPAWN"})
    assert len(model.text_stack.layers) == initial_count + 1, \
        f"Expected {initial_count + 1} layers after SPAWN, got {len(model.text_stack.layers)}"


def test_neurogenesis_prune_removes_layer():
    """Verify that PRUNE action physically removes a layer from the stack."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=3, layers_img=1, layers_audio=1
    )
    model = PCEIModel(config=config)
    controller = NeurogenesisController(model)
    initial_count = len(model.text_stack.layers)
    
    first_layer = model.text_stack.layers[0]
    if hasattr(first_layer, 'module'):
        layer_id = first_layer.module.layer_id
    else:
        layer_id = first_layer.layer_id
    
    controller.apply_structural_changes({layer_id: "PRUNE"})
    assert len(model.text_stack.layers) == initial_count - 1, \
        f"Expected {initial_count - 1} layers after PRUNE, got {len(model.text_stack.layers)}"


def test_neurogenesis_prune_preserves_minimum():
    """Verify pruning doesn't remove the last layer in a stack."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=1, layers_img=1, layers_audio=1
    )
    model = PCEIModel(config=config)
    controller = NeurogenesisController(model)
    
    first_layer = model.text_stack.layers[0]
    if hasattr(first_layer, 'module'):
        layer_id = first_layer.module.layer_id
    else:
        layer_id = first_layer.layer_id
    
    controller.apply_structural_changes({layer_id: "PRUNE"})
    assert len(model.text_stack.layers) >= 1, "Pruning removed the last layer!"


def test_neurogenesis_spawn_respects_max_layers():
    """Verify SPAWN is blocked at max capacity."""
    config = PCEIConfig(
        vocab_size=100, d_model_text=64, d_model_img=64, d_model_audio=64,
        layers_text=2, layers_img=1, layers_audio=1
    )
    model = PCEIModel(config=config)
    controller = NeurogenesisController(model, max_layers_per_stack=2)
    
    first_layer = model.text_stack.layers[0]
    if hasattr(first_layer, 'module'):
        layer_id = first_layer.module.layer_id
    else:
        layer_id = first_layer.layer_id
    
    controller.apply_structural_changes({layer_id: "SPAWN"})
    assert len(model.text_stack.layers) == 2, "SPAWN should be blocked at max capacity"

