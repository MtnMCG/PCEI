with open('tools/instruction_tuner.py', 'r') as f:
    code = f.read()

code = code.replace("print('Before forward')\n            logits, t_latent, _, _, _, _ = nexus.core(", "logits, t_latent, _, _, _, _ = nexus.core(")
code = code.replace("print('After forward, before backward')\n            total_loss = loss", "total_loss = loss")
code = code.replace("total_loss.backward()\n            print('After backward, before step')", "total_loss.backward()")
code = code.replace("optimizer.step()\n            print('After step')", "optimizer.step()")

with open('tools/instruction_tuner.py', 'w') as f:
    f.write(code)
print("Cleaned prints")
