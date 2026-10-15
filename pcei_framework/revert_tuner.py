with open('tools/instruction_tuner.py', 'r') as f:
    code = f.read()

code = code.replace("""
import torch.nn as nn
def hook_fn(module, input, output):
    if hasattr(module, 'weight'):
        if input[0].dtype != module.weight.dtype:
            print(f"CRASH IMMINENT in {module}: input {input[0].dtype}, weight {module.weight.dtype}")
            import sys
            sys.stdout.flush()

for m in nexus.modules():
    if isinstance(m, nn.Linear):
        m.register_forward_hook(hook_fn)
""", "")

with open('tools/instruction_tuner.py', 'w') as f:
    f.write(code)
print("Reverted")
