import torch
ckpt_path = "checkpoints/pcei_micro.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
for k in checkpoint['model_state_dict'].keys():
    if "core.text_stack.layers" in k:
        print(k)
