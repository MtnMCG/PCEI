import os
import glob

for f_path in glob.glob('tools/*.py'):
    with open(f_path, 'r') as f:
        code = f.read()
    
    modified = False

    search_str = "optimizer.load_state_dict(ckpt['optimizer_state_dict'])"
    replace_str = """optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    # Force all optimizer states to bfloat16 for MPS compatibility
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(nexus.system_dtype)"""
    
    if search_str in code and replace_str not in code:
        code = code.replace(search_str, replace_str)
        modified = True

    if modified:
        with open(f_path, 'w') as f:
            f.write(code)
        print(f"Patched optimizer loading in {f_path}")

print("Done patching optimizer states")
