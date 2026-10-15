import re

with open('src/core/ingestors.py', 'r') as f:
    code = f.read()

# For VideoIngestor methods
code = re.sub(r'pad = torch.zeros\(target_patches - stacked.shape\[0\], stacked.shape\[1\]\)', 
              r'pad = torch.zeros(target_patches - stacked.shape[0], stacked.shape[1], dtype=stacked.dtype)', code)

code = re.sub(r"results\['image'\] = torch.zeros\(1, self.max_frames \* self.image_ingestor.num_patches, self.image_ingestor.patch_dim\)", 
              r"results['image'] = torch.zeros(1, self.max_frames * self.image_ingestor.num_patches, self.image_ingestor.patch_dim, dtype=self.dtype)", code)

code = re.sub(r"results\['audio'\] = torch.zeros\(1, 100, 80\)", 
              r"results['audio'] = torch.zeros(1, 100, 80, dtype=self.dtype)", code)

code = re.sub(r"pad = torch.zeros\(target_patches - all_patches.shape\[0\], all_patches.shape\[1\]\)", 
              r"pad = torch.zeros(target_patches - all_patches.shape[0], all_patches.shape[1], dtype=all_patches.dtype)", code)

# For dataset methods
code = re.sub(r"pad = torch.zeros\(target_len - mel.shape\[0\], mel.shape\[1\]\)", 
              r"pad = torch.zeros(target_len - mel.shape[0], mel.shape[1], dtype=mel.dtype)", code)

code = re.sub(r"pad = torch.zeros\(img_target_len - ti.shape\[0\], ti.shape\[1\]\)", 
              r"pad = torch.zeros(img_target_len - ti.shape[0], ti.shape[1], dtype=ti.dtype)", code)

code = re.sub(r"batch_image.append\(torch.zeros\(img_target_len, self.image_ingestor.patch_dim\)\)", 
              r"batch_image.append(torch.zeros(img_target_len, self.image_ingestor.patch_dim, dtype=self.image_ingestor.dtype))", code)

code = re.sub(r"pad = torch.zeros\(audio_target_len - mel.shape\[0\], mel.shape\[1\]\)", 
              r"pad = torch.zeros(audio_target_len - mel.shape[0], mel.shape[1], dtype=mel.dtype)", code)

code = re.sub(r"batch_audio.append\(torch.zeros\(audio_target_len, 80\)\)", 
              r"batch_audio.append(torch.zeros(audio_target_len, 80, dtype=self.audio_ingestor.dtype))", code)

with open('src/core/ingestors.py', 'w') as f:
    f.write(code)

print("Patched ingestors.py successfully.")
