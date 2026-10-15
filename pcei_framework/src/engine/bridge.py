"""
PCEI Weight Bridge — Real-Time State Synchronization

Allows a 'Witness' process (like chat_live.py) to automatically synchronize 
with the 'Training' process (train.py) by monitoring checkpoint updates.
"""
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

class WeightBridge:
    """
    Monitors a specific checkpoint file and hot-loads weights into a model 
    whenever the file is updated on disk.
    """
    def __init__(self, model: nn.Module, checkpoint_path: Union[str, Path]):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.last_mtime = 0
        self.total_syncs = 0
        
        # Initial check
        if self.checkpoint_path.exists():
            self.last_mtime = self.checkpoint_path.stat().st_mtime
            
    def check_and_sync(self, force: bool = False) -> bool:
        """
        Checks if the checkpoint has been updated.
        If yes (or if force=True), loads state_dict into the model.
        Returns True if a sync occurred.
        """
        if not self.checkpoint_path.exists():
            return False
            
        current_mtime = self.checkpoint_path.stat().st_mtime
        
        if force or current_mtime > self.last_mtime:
            try:
                # Load with map_location to ensure it matches current device
                device = next(self.model.parameters()).device
                checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                
                # Support both full checkpoints and state_dicts
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                self.model.load_state_dict(state_dict)
                self.last_mtime = current_mtime
                self.total_syncs += 1
                return True
            except Exception as e:
                print(f"[WeightBridge] Sync error: {e}")
                return False
                
        return False

    def auto_sync_loop(self, interval: float = 1.0):
        """Simple blocking loop for background synchronization (use in a thread)."""
        while True:
            if self.check_and_sync():
                print(f"[WeightBridge] Hot-swapped weights (Sync #{self.total_syncs})")
            time.sleep(interval)
