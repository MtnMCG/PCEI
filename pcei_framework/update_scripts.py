import re
from pathlib import Path

files_to_update = [
    "tools/dreaming_chamber.py",
    "tools/resonance_chat.py",
    "tools/multimodal_tuner.py",
    "tools/wiki_explorer.py",
    "scripts/train.py"
]

for file_path in files_to_update:
    path = Path("/Users/emerson/Desktop/PCEI /pcei_framework") / file_path
    if not path.exists(): continue
    
    with open(path, 'r') as f:
        content = f.read()
        
    # 1. Insert config loading
    if "from src.config import PCEIConfig" not in content:
        # Find checkpoint_path or ckpt_path assignment
        content = re.sub(
            r'([ \t]+)(c[hec]*k[poin]*t_path = [^\n]+)',
            r'\1from src.config import PCEIConfig\n\1config = PCEIConfig.load()\n\1\2',
            content, count=1
        )
        
    # 2. Replace load_topology_from_checkpoint arguments
    content = re.sub(
        r'PCEINexus\.load_topology_from_checkpoint\([^)]+\)',
        r'PCEINexus.load_topology_from_checkpoint(\n        ckpt_path if "ckpt_path" in locals() else checkpoint_path,\n        config.layers_text, config.layers_img, config.layers_audio,\n        config.d_model_text, config.d_model_img, config.d_model_audio\n    )',
        content
    )
    
    # 3. Replace PCEINexus(...) args
    content = re.sub(
        r'nexus = PCEINexus\([^)]+\)',
        r'nexus = PCEINexus(\n        config=config,\n        device=device\n    )',
        content
    )
    
    with open(path, 'w') as f:
        f.write(content)
print("Scripts updated.")
