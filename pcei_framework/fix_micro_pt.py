import os
import glob

for f_path in glob.glob('tools/*.py'):
    with open(f_path, 'r') as f:
        code = f.read()
    
    if "pcei_micro.pt" in code:
        code = code.replace("pcei_micro.pt", "pcei_100m.pt")
        with open(f_path, 'w') as f:
            f.write(code)
        print(f"Patched pcei_micro.pt in {f_path}")

print("Done patching pcei_micro.pt")
