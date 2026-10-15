import torch
import sys

for ckpt in ["checkpoints/pcei_checkpoint_stable.pt", "checkpoints/pcei_checkpoint_remediated_v4.pt", "checkpoints/pcei_checkpoint.pt"]:
    try:
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        nan_keys = []
        for k, v in state_dict.items():
            if torch.isnan(v).any():
                nan_keys.append(k)
        print(f"[{ckpt}] NaNs: {len(nan_keys)}")
    except Exception as e:
        pass
