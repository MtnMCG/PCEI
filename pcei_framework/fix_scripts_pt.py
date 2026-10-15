import os
import glob
import re

for f_path in glob.glob('scripts/*.py'):
    with open(f_path, 'r') as f:
        code = f.read()
    
    modified = False
    
    # Replace all known old checkpoints with pcei_100m.pt
    checkpoints = ["pcei_1b.pt", "pcei_micro.pt", "pcei_500m.pt"]
    for ckpt in checkpoints:
        if ckpt in code:
            code = code.replace(ckpt, "pcei_100m.pt")
            modified = True
            
    if modified:
        with open(f_path, 'w') as f:
            f.write(code)
        print(f"Patched checkpoints in {f_path}")

print("Done patching scripts/")
