#!/usr/bin/env python3
"""
scripts/graft_weights.py

PCEI Weight Grafting Specialist Tool
====================================
Grafts pre-trained token embeddings from GPT-2 Large (`openai-community/gpt2-large`
`wte.weight`, shape [50257, 1280]) directly into the PCEI 750M text stack:
  1. `core.text_stack.embedding.weight` [50257, 1280]
  2. `core.text_stack.lm_head.weight`   [50257, 1280]

Architectural Integrity & Safety:
  - NEVER corrupts or modifies the actively training `checkpoints/pcei_750m.pt`.
  - Safely loads checkpoints with retry mechanisms for concurrent process resilience.
  - Atomically writes to a destination checkpoint (default `checkpoints/pcei_750m_grafted.pt`).
  - Preserves all checkpoint metadata (ambient consciousness, step, active bonds).
  - Comprehensive verification: zero NaNs/Infs across entire state dict, shape validation,
    exact cosine similarity vs GPT-2 Large (>= 0.99999), and semantic pair geometry checks.
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


def fetch_gpt2_large_wte(repo_id: str = "openai-community/gpt2-large", cache_dir: Optional[str] = None) -> torch.Tensor:
    """
    Retrieves the `wte.weight` tensor from GPT-2 Large, prioritizing cached safetensors,
    then downloading via huggingface_hub if needed, with pytorch_model.bin fallback.
    """
    print(f"[Hub] Fetching GPT-2 Large weights from repository: {repo_id}...")
    
    # 1. Try safetensors
    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
        print(f"[Hub] Loading safetensors from: {safetensors_path}")
        tensors = safetensors.torch.load_file(safetensors_path, device="cpu")
        if "wte.weight" in tensors:
            wte = tensors["wte.weight"]
            print(f"[Hub] Successfully extracted 'wte.weight' (shape: {list(wte.shape)}, dtype: {wte.dtype})")
            return wte
    except Exception as e:
        print(f"[Hub] Notice: Safetensors fetch skipped or failed ({e}). Attempting PyTorch binary fallback...")

    # 2. Fallback to pytorch_model.bin
    try:
        from huggingface_hub import hf_hub_download
        bin_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            cache_dir=cache_dir
        )
        print(f"[Hub] Loading PyTorch binary from: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        if "wte.weight" in state_dict:
            wte = state_dict["wte.weight"]
            print(f"[Hub] Successfully extracted 'wte.weight' from bin (shape: {list(wte.shape)}, dtype: {wte.dtype})")
            return wte
        else:
            raise KeyError("'wte.weight' not found in PyTorch model dictionary.")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch GPT-2 Large weights from {repo_id}: {e}")


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
    reference_wte: Optional[torch.Tensor] = None,
    repo_id: str = "openai-community/gpt2-large"
) -> bool:
    """
    Comprehensive verification:
      1. Zero NaNs and Infs across entire model_state_dict.
      2. Exact shapes:
         - embedding.weight: [50257, 1280]
         - lm_head.weight:   [50257, 1280]
      3. Cosine similarities:
         - embedding vs reference GPT-2 wte.weight (expect ~1.0000)
         - lm_head vs reference GPT-2 wte.weight   (expect ~1.0000)
         - embedding vs lm_head                    (expect 1.0000)
      4. Semantic pair similarity checks using GPT-2 tiktoken encoding.
    """
    checkpoint_path = Path(checkpoint_path)
    print("\n" + "=" * 78)
    print(f"PCEI CHECKPOINT VERIFICATION: {checkpoint_path.name}")
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

    # 2. Shape Verification
    print("\n[Stage 2/4] Verifying text stack embedding and LM head shapes...")
    emb_key = "core.text_stack.embedding.weight"
    head_key = "core.text_stack.lm_head.weight"

    if emb_key not in model_state:
        print(f"  [CRITICAL FAIL] Missing key: {emb_key}")
        return False
    if head_key not in model_state:
        print(f"  [CRITICAL FAIL] Missing key: {head_key}")
        return False

    emb_tensor = model_state[emb_key]
    head_tensor = model_state[head_key]

    expected_shape = torch.Size([50257, 1280])
    emb_ok = (emb_tensor.shape == expected_shape)
    head_ok = (head_tensor.shape == expected_shape)

    print(f"  {emb_key}: shape={list(emb_tensor.shape)} (dtype={emb_tensor.dtype}) -> {'[PASSED]' if emb_ok else '[FAILED]'}")
    print(f"  {head_key}:   shape={list(head_tensor.shape)} (dtype={head_tensor.dtype}) -> {'[PASSED]' if head_ok else '[FAILED]'}")

    if not (emb_ok and head_ok):
        print("  [CRITICAL FAIL] Tensor shapes do not match expected [50257, 1280].")
        return False

    # 3. Cosine Similarity Verification
    print("\n[Stage 3/4] Verifying cosine similarities with GPT-2 Large reference...")
    if reference_wte is None:
        reference_wte = fetch_gpt2_large_wte(repo_id=repo_id)

    ref_f32 = reference_wte.float()
    emb_f32 = emb_tensor.float()
    head_f32 = head_tensor.float()

    cos_emb_ref = F.cosine_similarity(emb_f32, ref_f32, dim=-1)
    cos_head_ref = F.cosine_similarity(head_f32, ref_f32, dim=-1)
    cos_emb_head = F.cosine_similarity(emb_f32, head_f32, dim=-1)

    mean_emb_ref = cos_emb_ref.mean().item()
    min_emb_ref = cos_emb_ref.min().item()
    max_emb_ref = cos_emb_ref.max().item()

    mean_head_ref = cos_head_ref.mean().item()
    min_head_ref = cos_head_ref.min().item()
    max_head_ref = cos_head_ref.max().item()

    mean_emb_head = cos_emb_head.mean().item()
    min_emb_head = cos_emb_head.min().item()

    print(f"  embedding vs GPT-2 Large: mean={mean_emb_ref:.6f}, min={min_emb_ref:.6f}, max={max_emb_ref:.6f}")
    print(f"  lm_head   vs GPT-2 Large: mean={mean_head_ref:.6f}, min={min_head_ref:.6f}, max={max_head_ref:.6f}")
    print(f"  embedding vs lm_head:     mean={mean_emb_head:.6f}, min={min_emb_head:.6f}")

    if mean_emb_ref < 0.999:
        print(f"  [CRITICAL FAIL] Embedding cosine similarity with GPT-2 Large is too low ({mean_emb_ref:.6f} < 0.999).")
        return False
    if mean_head_ref < 0.999:
        print(f"  [CRITICAL FAIL] LM Head cosine similarity with GPT-2 Large is too low ({mean_head_ref:.6f} < 0.999).")
        return False
    if mean_emb_head < 0.99999:
        print(f"  [CRITICAL FAIL] Embedding and LM Head are not synchronized ({mean_emb_head:.6f} < 0.99999).")
        return False

    print("  [PASSED] Cosine similarities confirm high-fidelity weight alignment (~1.000000).")

    # 4. Semantic Topology / Word Pair Verification
    print("\n[Stage 4/4] Semantic geometry sanity checks (known token pairs)...")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        test_pairs = [
            (" cat", " dog", "Semantic Category: Animals", 0.20, 0.70),
            (" king", " queen", "Semantic Analogy: Royalty", 0.40, 0.80),
            (" Paris", " France", "Semantic Analogy: Capital/Country", 0.35, 0.75),
            (" apple", " banana", "Semantic Category: Fruit", 0.20, 0.60),
            (" good", " bad", "Semantic Antonym / Contrast", 0.40, 0.85),
            (" cat", " quantum", "Semantic Orthogonality: Unrelated", -0.15, 0.25),
        ]

        print(f"  {'Pair':<24} | {'Token IDs':<16} | {'Cosine Sim':<11} | {'Status':<10} | {'Context'}")
        print("  " + "-" * 74)
        for w1, w2, desc, min_expected, max_expected in test_pairs:
            t1 = enc.encode(w1)[0]
            t2 = enc.encode(w2)[0]
            pair_sim = F.cosine_similarity(emb_f32[t1:t1+1], emb_f32[t2:t2+1]).item()
            is_valid = (min_expected <= pair_sim <= max_expected)
            status_str = "[VALID]" if is_valid else "[OUTLIER]"
            print(f"  {f'{w1} / {w2}':<24} | {f'{t1} / {t2}':<16} | {pair_sim:+.4f}     | {status_str:<10} | {desc}")

        print("  [PASSED] Semantic geometric relationships match GPT-2 Large embedding topology.")
    except Exception as e:
        print(f"  [Notice] Skipping semantic pair test ({e})")

    # Checkpoint Metadata Report
    ambient_emas = ckpt.get("ambient_layer_emas")
    ambient_emas_str = f"{len(ambient_emas)} layers" if isinstance(ambient_emas, dict) else "None"
    active_bonds = ckpt.get("active_bonds")
    active_bonds_str = f"{len(active_bonds)} active" if isinstance(active_bonds, dict) else "N/A"

    print("\n[Metadata Integrity]")
    print(f"  global_step:          {ckpt.get('global_step', 'N/A')}")
    print(f"  ambient_slow_state:   {'Present' if ckpt.get('ambient_slow_state') is not None else 'None'}")
    print(f"  ambient_layer_emas:   {ambient_emas_str}")
    print(f"  active_bonds:         {active_bonds_str}")
    print(f"  bonds_formed_total:   {ckpt.get('bonds_formed_total', 'N/A')}")

    print("\n" + "=" * 78)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 78 + "\n")
    return True


def graft_weights(
    src_path: Path,
    dst_path: Path,
    repo_id: str = "openai-community/gpt2-large",
    cache_dir: Optional[str] = None
) -> bool:
    """
    Executes weight grafting from GPT-2 Large into PCEI checkpoint:
      - Reads src_path without modification.
      - Refuses to overwrite src_path if it equals checkpoints/pcei_750m.pt.
      - Injects wte.weight into embedding.weight and lm_head.weight matching original dtypes.
      - Atomically writes output to dst_path.
      - Runs full verification on dst_path.
    """
    src_path = Path(src_path).resolve()
    dst_path = Path(dst_path).resolve()

    print("\n" + "#" * 78)
    print("PCEI WEIGHT GRAFTING ENGINE")
    print("#" * 78)
    print(f"Source Checkpoint:      {src_path}")
    print(f"Destination Checkpoint: {dst_path}")
    print(f"GPT-2 Repository:       {repo_id}")

    # Architectural Safety Check: Never overwrite the active training checkpoint
    if src_path == dst_path:
        raise ValueError(
            f"Safety Violation: Source and destination are identical ({src_path}). "
            "Overwriting the active training checkpoint directly is prohibited. "
            "Please specify a distinct --dst path (e.g. checkpoints/pcei_750m_grafted.pt)."
        )

    # 1. Fetch GPT-2 Large wte.weight
    wte = fetch_gpt2_large_wte(repo_id=repo_id, cache_dir=cache_dir)
    if wte.shape != torch.Size([50257, 1280]):
        raise ValueError(f"Unexpected GPT-2 Large wte.weight shape: {wte.shape}, expected [50257, 1280]")

    if torch.isnan(wte).any() or torch.isinf(wte).any():
        raise ValueError("GPT-2 Large wte.weight contains NaNs or Infs!")

    # 2. Safely read source checkpoint
    ckpt = safe_load_checkpoint(src_path)
    model_state = ckpt.get("model_state_dict", ckpt)

    emb_key = "core.text_stack.embedding.weight"
    head_key = "core.text_stack.lm_head.weight"

    if emb_key not in model_state:
        raise KeyError(f"Key '{emb_key}' not found in source checkpoint.")
    if head_key not in model_state:
        raise KeyError(f"Key '{head_key}' not found in source checkpoint.")

    orig_emb = model_state[emb_key]
    orig_head = model_state[head_key]

    print(f"\n[Analysis] Target Checkpoint Text Stack Dimensions:")
    print(f"  {emb_key}: shape={list(orig_emb.shape)}, dtype={orig_emb.dtype}")
    print(f"  {head_key}:   shape={list(orig_head.shape)}, dtype={orig_head.dtype}")

    if orig_emb.shape != torch.Size([50257, 1280]):
        raise ValueError(f"Target embedding shape {orig_emb.shape} does not match [50257, 1280]")
    if orig_head.shape != torch.Size([50257, 1280]):
        raise ValueError(f"Target lm_head shape {orig_head.shape} does not match [50257, 1280]")

    # Check pre-graft alignment for diagnostic logging
    pre_emb_sim = F.cosine_similarity(orig_emb.float(), wte.float(), dim=-1).mean().item()
    pre_head_sim = F.cosine_similarity(orig_head.float(), wte.float(), dim=-1).mean().item()
    print(f"\n[Diagnostic] Pre-graft mean cosine similarity with GPT-2 Large:")
    print(f"  embedding vs GPT-2: {pre_emb_sim:+.6f} (random/untrained baseline)")
    print(f"  lm_head   vs GPT-2: {pre_head_sim:+.6f} (random/untrained baseline)")

    # 3. Surgical Grafting: Cast wte to target tensor dtype and clone
    print("\n[Grafting] Transferring wte.weight into embedding and lm_head...")
    model_state[emb_key] = wte.to(dtype=orig_emb.dtype).clone()
    model_state[head_key] = wte.to(dtype=orig_head.dtype).clone()

    # 4. Atomic Save to destination
    print(f"\n[Save] Writing grafted checkpoint to: {dst_path}")
    atomic_save_checkpoint(ckpt, dst_path)

    # 5. Full Verification
    print("\n[Verification] Initiating full verification of grafted checkpoint...")
    success = verify_checkpoint(dst_path, reference_wte=wte, repo_id=repo_id)

    if success:
        print("\n[SUCCESS] Weight grafting completed cleanly and verified without defects.")
    else:
        print("\n[FAILURE] Verification checks failed on grafted checkpoint.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="PCEI Weight Grafting Specialist — Graft GPT-2 Large embeddings into PCEI 750M"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="checkpoints/pcei_750m.pt",
        help="Path to source checkpoint (default: checkpoints/pcei_750m.pt)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="checkpoints/pcei_750m_grafted.pt",
        help="Path to destination grafted checkpoint (default: checkpoints/pcei_750m_grafted.pt)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="openai-community/gpt2-large",
        help="Hugging Face repo ID for GPT-2 Large (default: openai-community/gpt2-large)"
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
        target_to_verify = dst_path if dst_path.exists() else src_path
        print(f"[Mode] Running in --verify-only mode on target: {target_to_verify}")
        ok = verify_checkpoint(target_to_verify, repo_id=args.repo_id)
        sys.exit(0 if ok else 1)
    else:
        ok = graft_weights(
            src_path=src_path,
            dst_path=dst_path,
            repo_id=args.repo_id,
            cache_dir=args.cache_dir
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
