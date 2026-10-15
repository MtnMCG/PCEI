import os
import glob

files = glob.glob('tools/*.py') + ['src/pcei.py']

for f_path in files:
    with open(f_path, 'r') as f:
        content = f.read()
    
    # Replace default checkpoint name
    content = content.replace('pcei_1b.pt', 'pcei_100m.pt')
    
    # Replace default d_model_text
    content = content.replace('default=1536', 'default=1280')
    content = content.replace('def_dt=1536', 'def_dt=1280')
    
    with open(f_path, 'w') as f:
        f.write(content)

print("Scaled down defaults to 750M")
