import os
import time
import threading
from pathlib import Path
import torch
from fastapi import APIRouter

from src.config import PCEIConfig

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

_stats_cache = {}
_analysis_lock = threading.Lock()
CACHE_TTL = 10  # 10s cache TTL for responsiveness

def get_latest_checkpoint() -> Path:
    if not CHECKPOINTS_DIR.exists():
        return None
    ckpts = list(CHECKPOINTS_DIR.glob("*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)

def analyze_checkpoint():
    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        return None
        
    current_time = time.time()
    cache_key = f"{ckpt_path}:{os.path.getmtime(ckpt_path)}"
    
    if cache_key in _stats_cache:
        cache_entry = _stats_cache[cache_key]
        if current_time - cache_entry['time'] < CACHE_TTL:
            return cache_entry['data']
            
    with _analysis_lock:
        # Check cache again inside the lock in case another thread just populated it
        if cache_key in _stats_cache:
            cache_entry = _stats_cache[cache_key]
            if current_time - cache_entry['time'] < CACHE_TTL:
                return cache_entry['data']
                
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
        
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        state_dict = {}
        
    global_step = ckpt.get("global_step", 0) if isinstance(ckpt, dict) else 0
    
    total_params = 0
    trainable_params = 0
    
    text_layers, img_layers, audio_layers = set(), set(), set()
    dt, di, da = 0, 0, 0
    text_p, img_p, aud_p = 0, 0, 0
    
    ternary_neg, ternary_zero, ternary_pos = 0, 0, 0
    total_ternary = 0
    
    per_layer_confidence = []
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            params = tensor.numel()
            total_params += params
            trainable_params += params
            
            if "core.text_stack." in key or "text_stack." in key:
                text_p += params
                if "layers." in key:
                    text_layers.add(key.split("layers.")[1].split(".")[0])
            elif "core.img_stack." in key or "img_stack." in key:
                img_p += params
                if "layers." in key:
                    img_layers.add(key.split("layers.")[1].split(".")[0])
            elif "core.audio_stack." in key or "audio_stack." in key:
                aud_p += params
                if "layers." in key:
                    audio_layers.add(key.split("layers.")[1].split(".")[0])
                
            if "text_stack.layers.0.block.norm1.weight" in key: dt = tensor.shape[0]
            elif "img_stack.layers.0.block.norm1.weight" in key: di = tensor.shape[0]
            elif "audio_stack.layers.0.block.norm1.weight" in key: da = tensor.shape[0]
            
            if "slow_field_confidence" in key:
                layer_id = key.split(".slow_field_confidence")[0]
                conf = tensor.item() if tensor.numel() == 1 else 0.5
                px = state_dict.get(f"{layer_id}.pos_x", torch.tensor(0.0)).item() if f"{layer_id}.pos_x" in state_dict else 0.0
                py = state_dict.get(f"{layer_id}.pos_y", torch.tensor(0.0)).item() if f"{layer_id}.pos_y" in state_dict else 0.0
                pz = state_dict.get(f"{layer_id}.pos_z", torch.tensor(0.0)).item() if f"{layer_id}.pos_z" in state_dict else 0.0
                stack = "TEXT"
                if "img_stack" in layer_id: stack = "IMAGE"
                elif "audio_stack" in layer_id: stack = "AUDIO"
                per_layer_confidence.append({
                    "layer_id": layer_id,
                    "confidence": conf,
                    "effective_lr": 4e-4 * (1.0 - conf),
                    "stack": stack,
                    "position": {"x": px, "y": py, "z": pz}
                })
            
            if "weight" in key:
                if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    zeros = (tensor.abs() < 0.1).sum().item()
                    negs = (tensor < -0.1).sum().item()
                    pos = (tensor > 0.1).sum().item()
                    ternary_zero += zeros
                    ternary_neg += negs
                    ternary_pos += pos
                    total_ternary += (zeros + negs + pos)

    fast_norm = 0.0
    slow_norm = 0.0
    div = 0.0
    
    if isinstance(ckpt, dict):
        fast_state = ckpt.get("ambient_fast_state", None)
        slow_state = ckpt.get("ambient_slow_state", None)
        if fast_state is not None and torch.is_tensor(fast_state):
            fast_norm = fast_state.float().norm().item()
        if slow_state is not None and torch.is_tensor(slow_state):
            slow_norm = slow_state.float().norm().item()
        if fast_state is not None and slow_state is not None and torch.is_tensor(fast_state) and torch.is_tensor(slow_state):
            div = torch.norm(fast_state.float() - slow_state.float()).item()

    config = PCEIConfig.load(str(PROJECT_ROOT / "config.yaml"))
    file_size_bytes = os.path.getsize(ckpt_path)
    
    active_bonds_list = []
    topo_nodes = []
    for idx, c in enumerate(per_layer_confidence):
        topo_nodes.append({
            "id": c["layer_id"],
            "x": 0.1 + (idx * 0.15) % 0.8,
            "y": 0.2 + (idx * 0.1) % 0.6,
            "z": c["position"]["z"],
            "stack": c["stack"],
            "confidence": c["confidence"]
        })
        
    topo_bonds = []
    if isinstance(ckpt, dict) and "active_bonds" in ckpt and isinstance(ckpt["active_bonds"], dict):
        for k, v in ckpt["active_bonds"].items():
            if isinstance(k, (tuple, list)) and len(k) == 2:
                active_bonds_list.append({"layer_a": k[0], "layer_b": k[1], "strength": v})
                topo_bonds.append({"source": k[0], "target": k[1], "strength": v})
            
    stats = {
        "model_stats": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "stacks": [
                {"name": "TextStack", "layers": max(len(text_layers), config.layers_text), "d_model": dt or config.d_model_text, "params": text_p},
                {"name": "ImageStack", "layers": max(len(img_layers), config.layers_img), "d_model": di or config.d_model_img, "params": img_p},
                {"name": "AudioStack", "layers": max(len(audio_layers), config.layers_audio), "d_model": da or config.d_model_audio, "params": aud_p}
            ],
            "ternary_distribution": {
                "zero_pct": (ternary_zero / total_ternary) * 100 if total_ternary > 0 else 50,
                "one_pct": (ternary_pos / total_ternary) * 100 if total_ternary > 0 else 25,
                "two_pct": (ternary_neg / total_ternary) * 100 if total_ternary > 0 else 25,
            },
            "weight_distribution": {
                "neg": (ternary_neg / total_ternary) * 100 if total_ternary > 0 else 25,
                "zero": (ternary_zero / total_ternary) * 100 if total_ternary > 0 else 50,
                "pos": (ternary_pos / total_ternary) * 100 if total_ternary > 0 else 25,
            },
            "system_dtype": config.system_dtype,
            "dtype": config.system_dtype,
            "vocab_size": config.vocab_size,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "checkpoint": {
                "filename": ckpt_path.name,
                "size": file_size_bytes,
                "step": global_step
            },
            "checkpoint_info": {
                "filename": ckpt_path.name,
                "size_mb": file_size_bytes / (1024 * 1024),
                "last_modified": os.path.getmtime(ckpt_path),
                "global_step": global_step
            }
        },
        "ambient_stats": {
            "per_layer_confidence": per_layer_confidence,
            "layer_confidence": [c["confidence"] for c in per_layer_confidence] if per_layer_confidence else [0.5] * 24,
            "effective_lr": [c["effective_lr"] for c in per_layer_confidence] if per_layer_confidence else [4e-4] * 24,
            "fast_state_norm": fast_norm,
            "fast_norm": fast_norm,
            "slow_state_norm": slow_norm,
            "slow_norm": slow_norm,
            "fast_slow_divergence": div,
            "divergence": div
        },
        "entanglement": {
            "active": len(active_bonds_list),
            "total_formed": ckpt.get("bonds_formed_total", len(active_bonds_list)) if isinstance(ckpt, dict) else len(active_bonds_list),
            "total_dissolved": ckpt.get("bonds_dissolved_total", 0) if isinstance(ckpt, dict) else 0,
            "active_bonds": active_bonds_list,
            "bonds_formed_total": ckpt.get("bonds_formed_total", len(active_bonds_list)) if isinstance(ckpt, dict) else len(active_bonds_list),
            "bonds_dissolved_total": ckpt.get("bonds_dissolved_total", 0) if isinstance(ckpt, dict) else 0,
            "topology": {
                "nodes": topo_nodes,
                "bonds": topo_bonds
            }
        }
    }
    
    _stats_cache[cache_key] = {
        "time": current_time,
        "data": stats
    }
    
    # Force garbage collection to free the 3GB checkpoint from memory immediately
    del ckpt
    del state_dict
    import gc
    gc.collect()
    
    return stats

@router.get("/model/stats")
def get_model_stats():
    stats = analyze_checkpoint()
    if stats:
        return stats["model_stats"]
    return {"error": "No checkpoint found"}

@router.get("/model/ambient")
def get_model_ambient():
    stats = analyze_checkpoint()
    if stats:
        return stats["ambient_stats"]
    return {"error": "No checkpoint found"}

@router.get("/model/entanglement")
def get_model_entanglement():
    stats = analyze_checkpoint()
    if stats:
        return stats["entanglement"]
    return {"error": "No checkpoint found"}

@router.get("/model/config")
def get_model_config():
    config = PCEIConfig.load(str(PROJECT_ROOT / "config.yaml"))
    return config.__dict__

@router.get("/checkpoints")
def list_checkpoints():
    if not CHECKPOINTS_DIR.exists():
        return []
    ckpts = []
    for f in CHECKPOINTS_DIR.glob("*.pt"):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            step = ckpt.get("global_step", 0) if isinstance(ckpt, dict) else 0
        except:
            step = 0
            
        ckpts.append({
            "filename": f.name,
            "modified": os.path.getmtime(f),
            "size": os.path.getsize(f),
            "step": step
        })
    ckpts.sort(key=lambda x: x["modified"], reverse=True)
    return ckpts

