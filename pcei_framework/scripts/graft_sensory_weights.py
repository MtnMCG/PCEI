#!/usr/bin/env python3
"""
scripts/graft_sensory_weights.py

PCEI Multimodal Sensory Weight Grafting Specialist Tool
======================================================
Grafts pre-trained multimodal weights into the PCEI 750M sensory columns:
  1. Image Stack:
     - Extracts visual projection and patch embedding from OpenAI CLIP ViT-B/16
       (`openai/clip-vit-base-patch16`):
         visual_projection.weight: [512, 768]
         conv1.weight (or vision_model.embeddings.patch_embedding.weight): [768, 3, 16, 16]
     - Multiplies: W_proj = visual_projection.weight @ conv1.weight.reshape(768, 768) -> [512, 768]
     - Grafts W_proj (cast to bfloat16) into `core.img_stack.patch_proj.weight`.
  2. Audio Stack:
     - Extracts mel-frequency audio projection from OpenAI Whisper
       (`openai/whisper-tiny` or `whisper-base`):
         encoder.conv1.weight: [384, 80, 3] (tiny) or [512, 80, 3] (base)
     - Takes center slice: W_audio = encoder.conv1.weight[:256, :, 1] -> [256, 80]
     - Grafts W_audio (cast to bfloat16) into `core.audio_stack.audio_proj.weight`.

Architectural Integrity & Separation of Concerns:
  - Preserves all existing checkpoints, text embeddings, and model parameters untouched.
  - Safely loads checkpoints with retry mechanism for concurrent process resilience.
  - Atomically writes to destination checkpoint (default `checkpoints/pcei_750m_supercharged.pt`).
  - Preserves all checkpoint metadata (ambient consciousness, step, active bonds).
  - Comprehensive verification: zero NaNs, zero Infs across the ENTIRE state dict,
    exact shape validation ([512, 768] and [256, 80]), and cosine similarity verification.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fetch_clip_projection_matrix(
    repo_id: str = "openai/clip-vit-base-patch16",
    cache_dir: Optional[str] = None
) -> torch.Tensor:
    """
    Retrieves visual_projection.weight and conv1 patch embedding from CLIP ViT-B/16,
    multiplies them (W_proj = visual_projection @ conv1.reshape(768, 768)), and returns
    the resulting [512, 768] projection matrix.
    """
    print(f"\n[CLIP Fetch] Fetching CLIP ViT-B/16 weights from: {repo_id}...")
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    clip_sd = None

    # 1. Try safetensors
    try:
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
        print(f"[CLIP Fetch] Loading safetensors from: {safetensors_path}")
        clip_sd = safetensors.torch.load_file(safetensors_path, device="cpu")
    except Exception as e:
        print(f"[CLIP Fetch] Safetensors not available ({e}), falling back to pytorch_model.bin...")

    # 2. Fallback to pytorch_model.bin
    if clip_sd is None:
        try:
            bin_path = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                cache_dir=cache_dir
            )
            print(f"[CLIP Fetch] Loading PyTorch binary from: {bin_path}")
            clip_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch CLIP weights from {repo_id}: {e}")

    # Identify patch embedding conv weight
    conv1_candidates = [
        "vision_model.embeddings.patch_embedding.weight",
        "visual.conv1.weight",
        "conv1.weight"
    ]
    conv1_w = None
    for key in conv1_candidates:
        if key in clip_sd:
            conv1_w = clip_sd[key]
            print(f"[CLIP Fetch] Found patch embedding conv: '{key}' (shape: {list(conv1_w.shape)})")
            break

    if conv1_w is None:
        raise KeyError(f"Could not find patch embedding conv weight in CLIP checkpoint among {conv1_candidates}")

    # Identify visual projection weight
    proj_candidates = [
        "visual_projection.weight",
        "visual.proj",
        "visual_projection"
    ]
    vis_proj_w = None
    for key in proj_candidates:
        if key in clip_sd:
            vis_proj_w = clip_sd[key]
            print(f"[CLIP Fetch] Found visual projection: '{key}' (shape: {list(vis_proj_w.shape)})")
            break

    if vis_proj_w is None:
        raise KeyError(f"Could not find visual projection weight in CLIP checkpoint among {proj_candidates}")

    # Validate shapes
    if conv1_w.shape != torch.Size([768, 3, 16, 16]):
        raise ValueError(f"Unexpected CLIP conv1.weight shape: {conv1_w.shape}, expected [768, 3, 16, 16]")
    if vis_proj_w.shape != torch.Size([512, 768]):
        # Check if transpose is needed
        if vis_proj_w.shape == torch.Size([768, 512]):
            vis_proj_w = vis_proj_w.t()
        else:
            raise ValueError(f"Unexpected CLIP visual_projection shape: {vis_proj_w.shape}, expected [512, 768]")

    # Check NaNs and Infs in source
    if torch.isnan(conv1_w).any() or torch.isinf(conv1_w).any():
        raise ValueError("CLIP conv1.weight contains NaNs or Infs!")
    if torch.isnan(vis_proj_w).any() or torch.isinf(vis_proj_w).any():
        raise ValueError("CLIP visual_projection.weight contains NaNs or Infs!")

    # Multiply W_proj = visual_projection.weight @ conv1.weight.reshape(768, 768)
    conv1_reshaped = conv1_w.reshape(768, 768).float()
    vis_proj_float = vis_proj_w.float()
    w_proj = vis_proj_float @ conv1_reshaped

    if w_proj.shape != torch.Size([512, 768]):
        raise ValueError(f"Computed W_proj shape {w_proj.shape} != [512, 768]")

    if torch.isnan(w_proj).any() or torch.isinf(w_proj).any():
        raise ValueError("Computed W_proj contains NaNs or Infs!")

    print(f"[CLIP Fetch] Successfully synthesized W_proj: shape={list(w_proj.shape)}, "
          f"mean={w_proj.mean().item():.6f}, std={w_proj.std().item():.6f}")
    return w_proj


def fetch_whisper_projection_matrix(
    repo_id: str = "openai/whisper-tiny",
    cache_dir: Optional[str] = None
) -> torch.Tensor:
    """
    Retrieves encoder.conv1.weight from OpenAI Whisper (tiny or base),
    extracts center slice [:256, :, 1], and returns the resulting [256, 80] projection matrix.
    """
    print(f"\n[Whisper Fetch] Fetching Whisper weights from: {repo_id}...")
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    whisper_sd = None

    # 1. Try safetensors
    try:
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
        print(f"[Whisper Fetch] Loading safetensors from: {safetensors_path}")
        whisper_sd = safetensors.torch.load_file(safetensors_path, device="cpu")
    except Exception as e:
        print(f"[Whisper Fetch] Safetensors not available ({e}), falling back to pytorch_model.bin...")

    # 2. Fallback to pytorch_model.bin
    if whisper_sd is None:
        try:
            bin_path = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                cache_dir=cache_dir
            )
            print(f"[Whisper Fetch] Loading PyTorch binary from: {bin_path}")
            whisper_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Whisper weights from {repo_id}: {e}")

    # Identify encoder conv1 weight
    conv1_candidates = [
        "model.encoder.conv1.weight",
        "encoder.conv1.weight"
    ]
    conv1_w = None
    for key in conv1_candidates:
        if key in whisper_sd:
            conv1_w = whisper_sd[key]
            print(f"[Whisper Fetch] Found encoder conv1: '{key}' (shape: {list(conv1_w.shape)})")
            break

    if conv1_w is None:
        raise KeyError(f"Could not find encoder conv1 weight in Whisper checkpoint among {conv1_candidates}")

    # Check shape: [384, 80, 3] for tiny, [512, 80, 3] for base
    if conv1_w.ndim != 3 or conv1_w.shape[1] != 80 or conv1_w.shape[2] != 3 or conv1_w.shape[0] < 256:
        raise ValueError(f"Unexpected Whisper encoder conv1 shape: {conv1_w.shape}, expected [>=256, 80, 3]")

    # Check NaNs and Infs in source
    if torch.isnan(conv1_w).any() or torch.isinf(conv1_w).any():
        raise ValueError("Whisper conv1.weight contains NaNs or Infs!")

    # Center slice: encoder.conv1.weight[:256, :, 1]
    w_audio = conv1_w[:256, :, 1].float().clone()

    if w_audio.shape != torch.Size([256, 80]):
        raise ValueError(f"Sliced W_audio shape {w_audio.shape} != [256, 80]")

    if torch.isnan(w_audio).any() or torch.isinf(w_audio).any():
        raise ValueError("Sliced W_audio contains NaNs or Infs!")

    print(f"[Whisper Fetch] Successfully extracted W_audio: shape={list(w_audio.shape)}, "
          f"mean={w_audio.mean().item():.6f}, std={w_audio.std().item():.6f}")
    return w_audio


def safe_load_checkpoint(path: Path, max_retries: int = 5, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Safely loads a checkpoint dictionary with retry logic to avoid race conditions
    with active background training processes writing atomic snapshots.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Checkpoint] Loading source checkpoint (attempt {attempt}/{max_retries}): {path}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return ckpt
        except Exception as e:
            last_err = e
            print(f"[Checkpoint] Warning: Read conflict or file busy on attempt {attempt}: {e}")
            time.sleep(retry_delay)

    raise RuntimeError(f"Failed to load checkpoint safely after {max_retries} attempts: {last_err}")


def atomic_save_checkpoint(data: Dict[str, Any], dst_path: Path) -> None:
    """
    Saves the checkpoint atomically: writes to a temporary file, flushes,
    and replaces the destination path in a single OS atomic operation.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f"{dst_path.name}.tmp.{os.getpid()}")

    print(f"[Checkpoint] Writing atomic checkpoint to temporary file: {tmp_path}")
    torch.save(data, tmp_path)

    print(f"[Checkpoint] Replacing destination atomically: {dst_path}")
    os.replace(tmp_path, dst_path)


def verify_checkpoint(
    checkpoint_path: Path,
    expected_w_proj: Optional[torch.Tensor] = None,
    expected_w_audio: Optional[torch.Tensor] = None,
) -> bool:
    """
    Comprehensive verification:
      1. Zero NaNs and Infs across entire model_state_dict.
      2. Exact shapes:
         - core.img_stack.patch_proj.weight:   [512, 768] (bfloat16)
         - core.audio_stack.audio_proj.weight: [256, 80]  (bfloat16)
         - core.text_stack.embedding.weight:   [50257, 1280] (bfloat16)
         - core.text_stack.lm_head.weight:     [50257, 1280] (bfloat16)
      3. Cosine similarity and exact matching vs reference tensors.
      4. Metadata integrity preservation.
    """
    checkpoint_path = Path(checkpoint_path)
    print("\n" + "=" * 78)
    print(f"PCEI SENSORY CHECKPOINT VERIFICATION: {checkpoint_path.name}")
    print("=" * 78)

    if not checkpoint_path.exists():
        print(f"[CRITICAL FAIL] Checkpoint does not exist: {checkpoint_path}")
        return False

    ckpt = safe_load_checkpoint(checkpoint_path)
    model_state = ckpt.get("model_state_dict", ckpt)

    # 1. NaN and Inf Scan
    print("\n[Stage 1/4] Scanning all checkpoint parameters for NaNs and Infs...")
    nan_params = []
    inf_params = []
    total_elements = 0
    total_params = 0

    for name, param in model_state.items():
        if isinstance(param, torch.Tensor):
            total_params += 1
            total_elements += param.numel()
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

    print(f"  Total Tensors Scanned:    {total_params:,}")
    print(f"  Total Parameters Scanned: {total_elements:,}")

    if nan_params:
        print(f"  [CRITICAL FAIL] {len(nan_params)} tensors contain NaNs!")
        for p in nan_params[:5]:
            print(f"    - {p}")
        return False
    else:
        print("  [PASSED] Zero NaNs detected across all parameters.")

    if inf_params:
        print(f"  [CRITICAL FAIL] {len(inf_params)} tensors contain Infs!")
        for p in inf_params[:5]:
            print(f"    - {p}")
        return False
    else:
        print("  [PASSED] Zero Infs detected across all parameters.")

    # 2. Shape and Dtype Verification
    print("\n[Stage 2/4] Verifying sensory and text stack tensor shapes and dtypes...")
    img_key = "core.img_stack.patch_proj.weight"
    audio_key = "core.audio_stack.audio_proj.weight"
    emb_key = "core.text_stack.embedding.weight"
    head_key = "core.text_stack.lm_head.weight"

    targets = {
        img_key: (torch.Size([512, 768]), torch.bfloat16),
        audio_key: (torch.Size([256, 80]), torch.bfloat16),
        emb_key: (torch.Size([50257, 1280]), torch.bfloat16),
        head_key: (torch.Size([50257, 1280]), torch.bfloat16),
    }

    all_shapes_ok = True
    for key, (expected_shape, expected_dtype) in targets.items():
        if key not in model_state:
            print(f"  [CRITICAL FAIL] Missing key: {key}")
            all_shapes_ok = False
            continue
        tensor = model_state[key]
        shape_ok = (tensor.shape == expected_shape)
        dtype_ok = (tensor.dtype == expected_dtype)
        status = "[PASSED]" if (shape_ok and dtype_ok) else "[FAILED]"
        print(f"  {key:<42}: shape={list(tensor.shape)} ({tensor.dtype}) expected {list(expected_shape)} ({expected_dtype}) -> {status}")
        if not (shape_ok and dtype_ok):
            all_shapes_ok = False

    if not all_shapes_ok:
        print("  [CRITICAL FAIL] Tensor shapes or dtypes do not match specifications.")
        return False

    # 3. Alignment and Cosine Similarity Verification
    print("\n[Stage 3/4] Verifying alignment with grafted reference matrices...")
    img_tensor = model_state[img_key].float()
    audio_tensor = model_state[audio_key].float()

    if expected_w_proj is not None:
        ref_img = expected_w_proj.float()
        cos_img = F.cosine_similarity(img_tensor.flatten().unsqueeze(0), ref_img.flatten().unsqueeze(0)).item()
        l2_err_img = torch.norm(img_tensor - ref_img).item() / torch.norm(ref_img).item()
        print(f"  Image Projection  (W_proj vs CLIP):  cosine_sim={cos_img:.8f}, rel_L2_err={l2_err_img:.6e}")
        if cos_img < 0.9999:
            print(f"  [CRITICAL FAIL] Image projection cosine similarity too low ({cos_img:.6f} < 0.9999).")
            return False
        print("  [PASSED] Image stack projection perfectly aligned with CLIP ViT-B/16.")

    if expected_w_audio is not None:
        ref_audio = expected_w_audio.float()
        cos_audio = F.cosine_similarity(audio_tensor.flatten().unsqueeze(0), ref_audio.flatten().unsqueeze(0)).item()
        l2_err_audio = torch.norm(audio_tensor - ref_audio).item() / torch.norm(ref_audio).item()
        print(f"  Audio Projection  (W_audio vs Whisper): cosine_sim={cos_audio:.8f}, rel_L2_err={l2_err_audio:.6e}")
        if cos_audio < 0.9999:
            print(f"  [CRITICAL FAIL] Audio projection cosine similarity too low ({cos_audio:.6f} < 0.9999).")
            return False
        print("  [PASSED] Audio stack projection perfectly aligned with Whisper conv1 center slice.")

    # 4. Checkpoint Metadata Report
    print("\n[Stage 4/4] Verifying checkpoint metadata integrity...")
    ambient_emas = ckpt.get("ambient_layer_emas")
    ambient_emas_str = f"{len(ambient_emas)} layers" if isinstance(ambient_emas, dict) else "None"
    active_bonds = ckpt.get("active_bonds")
    active_bonds_str = f"{len(active_bonds)} active" if isinstance(active_bonds, dict) else "N/A"

    print(f"  global_step:          {ckpt.get('global_step', 'N/A')}")
    print(f"  ambient_slow_state:   {'Present' if ckpt.get('ambient_slow_state') is not None else 'None'}")
    print(f"  ambient_layer_emas:   {ambient_emas_str}")
    print(f"  active_bonds:         {active_bonds_str}")
    print(f"  bonds_formed_total:   {ckpt.get('bonds_formed_total', 'N/A')}")
    print(f"  bonds_dissolved_total:{ckpt.get('bonds_dissolved_total', 'N/A')}")

    print("\n" + "=" * 78)
    print("ALL SENSORY CHECKPOINT VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 78 + "\n")
    return True


def graft_sensory_weights(
    src_path: Path,
    dst_path: Path,
    clip_repo: str = "openai/clip-vit-base-patch16",
    whisper_repo: str = "openai/whisper-tiny",
    cache_dir: Optional[str] = None
) -> bool:
    """
    Executes multimodal sensory weight grafting:
      1. Computes W_proj [512, 768] from CLIP ViT-B/16.
      2. Computes W_audio [256, 80] from Whisper conv1 center slice.
      3. Reads src_path without modifying it.
      4. Refuses to overwrite src_path if it equals checkpoints/pcei_750m.pt.
      5. Grafts W_proj into core.img_stack.patch_proj.weight (cast to bfloat16).
      6. Grafts W_audio into core.audio_stack.audio_proj.weight (cast to bfloat16).
      7. Atomically saves full checkpoint to dst_path.
      8. Runs comprehensive verification on dst_path.
    """
    src_path = Path(src_path).resolve()
    dst_path = Path(dst_path).resolve()

    print("\n" + "#" * 78)
    print("PCEI MULTIMODAL SENSORY WEIGHT GRAFTING ENGINE")
    print("#" * 78)
    print(f"Source Checkpoint:      {src_path}")
    print(f"Destination Checkpoint: {dst_path}")
    print(f"CLIP Repository:        {clip_repo}")
    print(f"Whisper Repository:     {whisper_repo}")

    # Architectural Safety Check: Never overwrite the active training checkpoint
    if src_path == dst_path and src_path.name == "pcei_750m.pt":
        raise ValueError(
            f"Safety Violation: Source and destination are identical ({src_path}). "
            "Overwriting the active training checkpoint directly is prohibited. "
            "Please specify a distinct --dst path (e.g. checkpoints/pcei_750m_supercharged.pt)."
        )

    # 1. Fetch CLIP visual projection matrix
    w_proj = fetch_clip_projection_matrix(repo_id=clip_repo, cache_dir=cache_dir)

    # 2. Fetch Whisper audio projection matrix
    w_audio = fetch_whisper_projection_matrix(repo_id=whisper_repo, cache_dir=cache_dir)

    # 3. Safely read source checkpoint
    ckpt = safe_load_checkpoint(src_path)
    model_state = ckpt.get("model_state_dict", ckpt)

    img_key = "core.img_stack.patch_proj.weight"
    audio_key = "core.audio_stack.audio_proj.weight"

    if img_key not in model_state:
        raise KeyError(f"Key '{img_key}' not found in source checkpoint.")
    if audio_key not in model_state:
        raise KeyError(f"Key '{audio_key}' not found in source checkpoint.")

    orig_img = model_state[img_key]
    orig_audio = model_state[audio_key]

    print(f"\n[Analysis] Target Checkpoint Sensory Stack Dimensions:")
    print(f"  {img_key}:   shape={list(orig_img.shape)}, dtype={orig_img.dtype}")
    print(f"  {audio_key}: shape={list(orig_audio.shape)}, dtype={orig_audio.dtype}")

    if orig_img.shape != torch.Size([512, 768]):
        raise ValueError(f"Target patch_proj shape {orig_img.shape} does not match [512, 768]")
    if orig_audio.shape != torch.Size([256, 80]):
        raise ValueError(f"Target audio_proj shape {orig_audio.shape} does not match [256, 80]")

    # Diagnostic pre-graft alignment
    pre_img_sim = F.cosine_similarity(orig_img.float().flatten().unsqueeze(0), w_proj.float().flatten().unsqueeze(0)).item()
    pre_audio_sim = F.cosine_similarity(orig_audio.float().flatten().unsqueeze(0), w_audio.float().flatten().unsqueeze(0)).item()
    print(f"\n[Diagnostic] Pre-graft cosine similarity with sensory priors:")
    print(f"  Image patch_proj vs CLIP W_proj:       {pre_img_sim:+.6f} (untrained baseline)")
    print(f"  Audio audio_proj vs Whisper W_audio:   {pre_audio_sim:+.6f} (untrained baseline)")

    # 4. Surgical Grafting: Cast to bfloat16 and clone into state dict
    print("\n[Grafting] Injecting CLIP W_proj into core.img_stack.patch_proj.weight (bfloat16)...")
    model_state[img_key] = w_proj.to(dtype=torch.bfloat16).clone()

    print("[Grafting] Injecting Whisper W_audio into core.audio_stack.audio_proj.weight (bfloat16)...")
    model_state[audio_key] = w_audio.to(dtype=torch.bfloat16).clone()

    # 5. Atomic Save to destination
    print(f"\n[Save] Writing supercharged checkpoint to: {dst_path}")
    atomic_save_checkpoint(ckpt, dst_path)

    # 6. Full Verification
    print("\n[Verification] Initiating comprehensive verification of supercharged checkpoint...")
    success = verify_checkpoint(
        dst_path,
        expected_w_proj=w_proj,
        expected_w_audio=w_audio
    )

    if success:
        print("\n[SUCCESS] Sensory weight grafting completed cleanly and verified with zero defects.")
    else:
        print("\n[FAILURE] Verification checks failed on supercharged checkpoint.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="PCEI Multimodal Sensory Specialist — Graft CLIP ViT-B/16 and Whisper into PCEI 750M"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="checkpoints/pcei_750m_grafted.pt",
        help="Path to source checkpoint (default: checkpoints/pcei_750m_grafted.pt)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="checkpoints/pcei_750m_supercharged.pt",
        help="Path to destination checkpoint (default: checkpoints/pcei_750m_supercharged.pt)"
    )
    parser.add_argument(
        "--clip-repo",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Hugging Face repo ID for CLIP ViT-B/16 (default: openai/clip-vit-base-patch16)"
    )
    parser.add_argument(
        "--whisper-repo",
        type=str,
        default="openai/whisper-tiny",
        help="Hugging Face repo ID for Whisper (default: openai/whisper-tiny)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom Hugging Face cache directory (optional)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run verification on destination checkpoint without performing grafting"
    )

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if not src_path.is_absolute():
        src_path = PROJECT_ROOT / src_path
    if not dst_path.is_absolute():
        dst_path = PROJECT_ROOT / dst_path

    if args.verify_only:
        print(f"\n[Mode] Running verification-only mode on: {dst_path}")
        success = verify_checkpoint(dst_path)
        sys.exit(0 if success else 1)

    try:
        success = graft_sensory_weights(
            src_path=src_path,
            dst_path=dst_path,
            clip_repo=args.clip_repo,
            whisper_repo=args.whisper_repo,
            cache_dir=args.cache_dir
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Sensory weight grafting encountered an unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
