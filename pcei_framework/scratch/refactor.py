import os
import glob
from pathlib import Path

# 1. Update src/pcei.py
pcei_path = Path("src/pcei.py")
content = pcei_path.read_text()
new_func = """    @classmethod
    def load_topology_from_checkpoint(cls, ckpt_path: Path, default_text=2, default_img=1, default_audio=1, def_dt=128, def_di=128, def_da=64):
        if not ckpt_path.exists():
            return default_text, default_img, default_audio, def_dt, def_di, def_da
            
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            t_layers, i_layers, a_layers = set(), set(), set()
            dt, di, da = def_dt, def_di, def_da
            
            for key in state_dict.keys():
                if "core.text_stack.layers." in key:
                    t_layers.add(key.split("core.text_stack.layers.")[1].split(".")[0])
                elif "core.img_stack.layers." in key:
                    i_layers.add(key.split("core.img_stack.layers.")[1].split(".")[0])
                elif "core.audio_stack.layers." in key:
                    a_layers.add(key.split("core.audio_stack.layers.")[1].split(".")[0])
                    
                if "core.text_stack.layers.0.block.norm1.weight" in key: dt = state_dict[key].shape[0]
                elif "core.img_stack.layers.0.block.norm1.weight" in key: di = state_dict[key].shape[0]
                elif "core.audio_stack.layers.0.block.norm1.weight" in key: da = state_dict[key].shape[0]
                    
            return max(len(t_layers), default_text), max(len(i_layers), default_img), max(len(a_layers), default_audio), dt, di, da
        except Exception as e:
            print(f"[TopologyLoader] Failed to read checkpoint: {e}")
            return default_text, default_img, default_audio, def_dt, def_di, def_da
"""
# Replace the old func
start_idx = content.find("    @classmethod")
end_idx = content.find("    def __init__")
content = content[:start_idx] + new_func + "\n" + content[end_idx:]
pcei_path.write_text(content)

# 2. Update all scripts
for script in glob.glob("tools/*.py") + glob.glob("scripts/*.py"):
    if not os.path.isfile(script): continue
    text = Path(script).read_text()
    if "load_topology_from_checkpoint" in text:
        text = text.replace(
            "layers_text, layers_img, layers_audio = PCEINexus.load_topology_from_checkpoint",
            "layers_text, layers_img, layers_audio, d_text, d_img, d_audio = PCEINexus.load_topology_from_checkpoint"
        )
        # Update constructor args
        text = text.replace("d_model_text=args.d_model_text", "d_model_text=d_text")
        text = text.replace("d_model_img=args.d_model_img", "d_model_img=d_img")
        text = text.replace("d_model_audio=args.d_model_audio", "d_model_audio=d_audio")
        
        text = text.replace("d_model_text=128", "d_model_text=d_text")
        text = text.replace("d_model_img=128", "d_model_img=d_img")
        text = text.replace("d_model_audio=64", "d_model_audio=d_audio")
        
        text = text.replace("Audio\")", "Audio | {d_text} D_Text\")")
        text = text.replace("Audio[/cyan]\")", "Audio | {d_text} D_Text[/cyan]\")")
        Path(script).write_text(text)

print("Refactored successfully!")
