import os
import glob

for f_path in glob.glob('tools/*.py'):
    with open(f_path, 'r') as f:
        code = f.read()
    
    modified = False

    # Fix save_tensor_as_image to force float32
    if 'def save_tensor_as_image' in code:
        if 'tensor = tensor.float()' not in code:
            code = code.replace(
                'def save_tensor_as_image(tensor: torch.Tensor, filepath: Path):',
                'def save_tensor_as_image(tensor: torch.Tensor, filepath: Path):\n    tensor = tensor.float()'
            )
            modified = True

    # Fix specific torch.zeros and torch.randn that might be missing dtype
    if 'intent = torch.zeros(batch_size, d_text, device=device)' in code:
        code = code.replace(
            'intent = torch.zeros(batch_size, d_text, device=device)',
            'intent = torch.zeros(batch_size, d_text, device=device, dtype=nexus.system_dtype)'
        )
        modified = True

    if 'img_latent = torch.randn(1, 4, d_img, device=device) * 0.5' in code:
        code = code.replace(
            'img_latent = torch.randn(1, 4, d_img, device=device) * 0.5',
            'img_latent = torch.randn(1, 4, d_img, device=device, dtype=nexus.system_dtype) * 0.5'
        )
        modified = True

    if 'resp_noise = torch.randn(1, 32, 128, device=device)' in code:
        code = code.replace(
            'resp_noise = torch.randn(1, 32, 128, device=device)',
            'resp_noise = torch.randn(1, 32, 128, device=device, dtype=nexus.system_dtype)'
        )
        modified = True
        
    if modified:
        with open(f_path, 'w') as f:
            f.write(code)
        print(f"Patched {f_path}")

print("Done patching tools")
