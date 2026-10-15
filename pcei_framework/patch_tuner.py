import re

with open('tools/instruction_tuner.py', 'r') as f:
    code = f.read()

code = code.replace("logits, t_latent, _, _, _, _ = nexus.core(", "print('Before forward')\n            logits, t_latent, _, _, _, _ = nexus.core(")
code = code.replace("total_loss = loss", "print('After forward, before backward')\n            total_loss = loss")
code = code.replace("total_loss.backward()", "total_loss.backward()\n            print('After backward, before step')")
code = code.replace("optimizer.step()", "optimizer.step()\n            print('After step')")

with open('tools/instruction_tuner.py', 'w') as f:
    f.write(code)

print("Patched tuner with debug prints")
