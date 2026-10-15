import yaml
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class PCEIConfig:
    # Text Stack
    d_model_text: int = 768
    layers_text: int = 12
    
    # Image Stack
    d_model_img: int = 512
    layers_img: int = 4
    
    # Audio Stack
    d_model_audio: int = 256
    layers_audio: int = 4
    
    # General
    vocab_size: int = 50257
    system_dtype: str = "bfloat16"
    
    # Training
    base_lr: float = 4e-4
    weight_decay: float = 0.01
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> "PCEIConfig":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        return cls()
        
    def save(self, path: str = "config.yaml"):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
