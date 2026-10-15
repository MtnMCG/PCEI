import os
import glob

for f_path in glob.glob('tools/*.py'):
    with open(f_path, 'r') as f:
        code = f.read()
    
    modified = False

    if 'optimizer.load_state_dict(' in code and 'state[k] = v.to(nexus.system_dtype)' not in code:
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if 'optimizer.load_state_dict(' in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + '# Force all optimizer states to bfloat16 for MPS compatibility')
                new_lines.append(indent + 'for state in optimizer.state.values():')
                new_lines.append(indent + '    for k, v in state.items():')
                new_lines.append(indent + '        if torch.is_tensor(v):')
                new_lines.append(indent + '            state[k] = v.to(nexus.system_dtype)')
                modified = True
        code = '\n'.join(new_lines)

    if modified:
        with open(f_path, 'w') as f:
            f.write(code)
        print(f"Patched optimizer loading in {f_path}")

print("Done patching optimizer states 2")
