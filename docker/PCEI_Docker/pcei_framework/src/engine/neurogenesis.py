import torch
import torch.nn as nn
from typing import Dict


from src.modules.plasticity import PlasticLayer
from src.core.stack import ModalityStack

class NeurogenesisController:
    """
    Manages the dynamic architecture of the PCEI Framework.
    Monitors the "metabolic stress" (gradient norms) of each existing layer.
    If a stack maintains high stress, it spawns a new layer to distribute the load.
    If a layer maintains zero stress and high confidence, it can be pruned.
    """
    def __init__(self, model: nn.Module, spawn_threshold: float = 5.0, prune_threshold: float = 0.01,
                 max_layers_per_stack: int = 16):
        self.model = model
        self.spawn_threshold = spawn_threshold
        self.prune_threshold = prune_threshold
        self.max_layers_per_stack = max_layers_per_stack
        
        # Track historical stress per layer to avoid spawning off single spikes
        self.stress_history: Dict[str, float] = {}
        self.history_momentum = 0.9

    def monitor_gradients(self):
        """
        Called after a backward pass. Calculates and stores the gradient norm 
        for each PlasticLayer.
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, PlasticLayer):
                    # Calculate total gradient norm for this layer
                    total_norm = 0.0
                    for p in module.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    # Update EMA of stress
                    if module.layer_id not in self.stress_history:
                        self.stress_history[module.layer_id] = total_norm
                    else:
                        current = self.stress_history[module.layer_id]
                        self.stress_history[module.layer_id] = current * self.history_momentum + total_norm * (1.0 - self.history_momentum)

    def evaluate_topology(self) -> Dict[str, str]:
        """
        Evaluates the stress history and returns a dictionary of actions to take.
        Returns: {layer_id: "SPAWN"} or {layer_id: "PRUNE"}
        
        In a full implementation, this actually modifies the nn.ModuleList in the stacks.
        Here we return the recommended structural changes.
        """
        actions = {}
        for layer_id, stress in self.stress_history.items():
            # Find the actual module to check confidence
            target_module = None
            for _, m in self.model.named_modules():
                if isinstance(m, PlasticLayer) and m.layer_id == layer_id:
                    target_module = m
                    break
                    
            if target_module is None:
                continue
                
            confidence = target_module.slow_field_confidence.item()
            
            # SPAWN logic: High stress despite reasonable confidence,
            # meaning the current capacity cannot solve the pattern.
            if stress > self.spawn_threshold:
                actions[layer_id] = "SPAWN"
                # Reset stress to prevent infinite spawning
                self.stress_history[layer_id] = 0.0
                
            # PRUNE logic: Extremely low stress AND high confidence (fully consolidated)
            # In purely metabolic PCEI, we prune to save inference compute.
            elif stress < self.prune_threshold and confidence > 0.95:
                # We don't want to prune the ONLY layer in a stack, handled by the caller.
                actions[layer_id] = "PRUNE"
                
        return actions

    def apply_structural_changes(self, actions: Dict[str, str]):
        """
        Executes the SPAWN or PRUNE commands by directly mutating the 
        ModalityStacks in the model.
        """
        for layer_id, action in actions.items():
            # Find the stack and layer index that owns this layer
            target_stack = None
            target_idx = None
            
            # Search all stacks in the model
            stacks = []
            for name, module in self.model.named_modules():
                if isinstance(module, ModalityStack):
                    stacks.append(module)
            
            for stack in stacks:
                for idx, layer in enumerate(stack.layers):
                    # Handle TTTAdapter-wrapped layers
                    inner = layer
                    if hasattr(layer, 'module') and isinstance(layer.module, PlasticLayer):
                        inner = layer.module
                    if isinstance(inner, PlasticLayer) and inner.layer_id == layer_id:
                        target_stack = stack
                        target_idx = idx
                        break
                if target_stack is not None:
                    break
            
            if target_stack is None:
                continue
            
            if action == "SPAWN":
                if len(target_stack.layers) >= self.max_layers_per_stack:
                    print(f"[Neurogenesis] SPAWN blocked for {layer_id}: stack at max capacity "
                          f"({self.max_layers_per_stack} layers)")
                    continue
                
                # Get parent layer properties
                parent = target_stack.layers[target_idx]
                inner_parent = parent
                if hasattr(parent, 'module') and isinstance(parent.module, PlasticLayer):
                    inner_parent = parent.module
                
                pos = inner_parent.get_position()
                new_id = f"{layer_id}_spawn_{len(target_stack.layers)}"
                new_layer = PlasticLayer(
                    layer_id=new_id,
                    d_model=inner_parent.d_model,
                    initial_pos=(pos[0], pos[1], pos[2] + 0.5)
                ).to(next(self.model.parameters()).device)
                
                # Initialize near identity to avoid disrupting existing information flow
                with torch.no_grad():
                    for p in new_layer.parameters():
                        p.mul_(0.01)
                
                # Insert after the stressed layer
                target_stack.layers.insert(target_idx + 1, new_layer)
                print(f"[Neurogenesis] SPAWNED {new_id} adjacent to {layer_id} "
                      f"(stack now has {len(target_stack.layers)} layers)")
                
            elif action == "PRUNE":
                if len(target_stack.layers) <= 1:
                    print(f"[Neurogenesis] PRUNE blocked for {layer_id}: cannot remove last layer in stack")
                    continue
                
                del target_stack.layers[target_idx]
                if layer_id in self.stress_history:
                    del self.stress_history[layer_id]
                print(f"[Neurogenesis] PRUNED {layer_id} "
                      f"(stack now has {len(target_stack.layers)} layers)")
