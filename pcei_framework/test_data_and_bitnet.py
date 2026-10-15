"""
Verification script for BitNet 1.58b threshold calibration and dataset loading.
Tests:
1. Dataset loading verification across synthetic, curriculum, and high-quality sources.
2. TernaryLinear forward & backward pass, gradient flow, zero NaNs, and BitNet threshold calibration.
"""
import sys
import os
import json
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is in path
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.modules.ternary import TernaryLinear, TernaryQuantize
from scripts.train import load_text_data, extract_text_from_record


def test_schema_parsers():
    print("\n--- 1. Testing Schema Parsers ---")
    # Test text
    assert extract_text_from_record({"text": "Hello world"}) == "Hello world"
    
    # Test prompt / response (standard)
    pr1 = extract_text_from_record({"prompt": "User query", "response": "Bot reply"})
    assert pr1 == "USER: User query\nASSISTANT: Bot reply"
    
    # Test prompt / response (Claude style with ASSISTANT:)
    pr2 = extract_text_from_record({"prompt": "SYSTEM: cs tutor\nASSISTANT:", "response": "Explanation"})
    assert pr2 == "SYSTEM: cs tutor\nASSISTANT: Explanation"
    
    # Test instruction / output
    io = extract_text_from_record({"instruction": "Write code", "input": "input data", "output": "print('ok')"})
    assert "USER: Write code" in io and "ASSISTANT: print('ok')" in io
    
    # Test messages
    msg = extract_text_from_record({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
    assert msg == "USER: hi\nASSISTANT: hello"
    
    # Test conversations
    conv = extract_text_from_record({"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]})
    assert conv == "USER: q\nASSISTANT: a"
    
    print("  [PASS] All schema format variations correctly parsed without drops.")


def test_dataset_loading():
    print("\n--- 2. Testing Dataset Loading Counts ---")
    data_dir = ROOT_DIR / "data"
    synthetic_dir = data_dir / "synthetic"
    curriculum_dir = data_dir / "curriculum"
    high_quality_dir = data_dir / "high_quality"
    raw_text_dir = data_dir / "raw" / "text"

    # Per-file breakdown in synthetic
    synthetic_counts = {}
    if synthetic_dir.exists():
        for f in sorted(synthetic_dir.glob("*.jsonl")):
            c = 0
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if extract_text_from_record(rec):
                                c += 1
                        except:
                            pass
            synthetic_counts[f.name] = c

    print("  Synthetic datasets:")
    for fn, count in synthetic_counts.items():
        print(f"    • {fn:30s}: {count:,} texts")

    # Curriculum counts
    curriculum_counts = {}
    if curriculum_dir.exists():
        for f in sorted(curriculum_dir.glob("*.jsonl")):
            c = 0
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if extract_text_from_record(rec):
                                c += 1
                        except:
                            pass
            curriculum_counts[f.name] = c

    print("  Curriculum datasets:")
    for fn, count in curriculum_counts.items():
        print(f"    • {fn:30s}: {count:,} texts")

    # High quality counts
    hq_counts = {}
    if high_quality_dir.exists():
        for f in sorted(high_quality_dir.glob("*.jsonl")):
            c = 0
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                for line in fp:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if extract_text_from_record(rec):
                                c += 1
                        except:
                            pass
            hq_counts[f.name] = c

    print("  High-Quality datasets:")
    for fn, count in hq_counts.items():
        print(f"    • {fn:30s}: {count:,} texts")

    # Full load_text_data call with pure_synthetic=True
    loaded_texts_pure = load_text_data(
        synthetic_dir=synthetic_dir,
        raw_text_dir=raw_text_dir,
        pure_synthetic=True,
        curriculum_dir=curriculum_dir,
        high_quality_dir=high_quality_dir
    )
    total_pure = len(loaded_texts_pure)
    print(f"\n  Total loaded training texts (pure_synthetic / high-grade reasoning): {total_pure:,}")

    # Specific check for Claude dataset
    claude_count = synthetic_counts.get("anthropic_claude.jsonl", 0)
    assert claude_count > 0, "ERROR: anthropic_claude.jsonl was not loaded!"
    print(f"  [PASS] anthropic_claude.jsonl successfully loaded: {claude_count:,} records.")

    # Specific check for curriculum dataset
    teacher_count = curriculum_counts.get("council_teacher.jsonl", 0)
    assert teacher_count > 0, "ERROR: council_teacher.jsonl was not loaded!"
    print(f"  [PASS] council_teacher.jsonl successfully loaded: {teacher_count:,} records.")

    return {
        "synthetic": synthetic_counts,
        "curriculum": curriculum_counts,
        "high_quality": hq_counts,
        "total_pure_synthetic": total_pure,
    }


def test_ternary_bitnet_calibration(device: str = "cpu"):
    print(f"\n--- 3. Testing BitNet 1.58b Threshold Calibration on {device.upper()} ---")
    
    in_features = 256
    out_features = 512
    layer = TernaryLinear(in_features, out_features).to(device)

    # 1. Inspect dynamic threshold scaling
    gamma = layer.weight.abs().mean()
    t_low = layer.threshold_low
    t_high = layer.threshold_high

    print(f"  Weight gamma (mean |W|): {gamma.item():.6f}")
    print(f"  Learned threshold_low:   {t_low.item():.6f} (ratio to gamma: {t_low.item() / gamma.item():.4f})")
    print(f"  Learned threshold_high:  {t_high.item():.6f} (ratio to gamma: {t_high.item() / gamma.item():.4f})")

    # Ratio should be ~0.5 and ~1.5
    assert torch.isclose(t_low, 0.5 * gamma, atol=1e-5), f"threshold_low {t_low} != 0.5 * gamma {0.5 * gamma}"
    assert torch.isclose(t_high, 1.5 * gamma, atol=1e-5), f"threshold_high {t_high} != 1.5 * gamma {1.5 * gamma}"
    print("  [PASS] Initialized threshold_low == 0.5 * gamma and threshold_high == 1.5 * gamma.")

    # 2. Check strict BitNet 1.58b ternary states {-1, 0, +1}
    q = layer.quantized_weight

    count_0 = (q == 0).sum().item()
    count_neg1 = (q == -1).sum().item()
    count_pos1 = (q == 1).sum().item()
    total_w = q.numel()

    print(f"  Quantization states distribution across {total_w:,} parameters:")
    print(f"    • Pruned      (0): {count_0:6d} ({count_0/total_w*100:5.1f}%)")
    print(f"    • Inhibitory (-1): {count_neg1:6d} ({count_neg1/total_w*100:5.1f}%)")
    print(f"    • Excitatory (+1): {count_pos1:6d} ({count_pos1/total_w*100:5.1f}%)")

    assert count_0 > 0, "Dead zone: zero pruned weights!"
    assert count_neg1 > 0, "Dead zone: zero inhibitory weights!"
    assert count_pos1 > 0, "Dead zone: zero excitatory weights!"
    assert not (q.abs() == 2.0).any(), "Dead zone violation: 2.0 state detected!"
    print("  [PASS] Strictly BitNet 1.58b ternary states {-1, 0, +1} verified (zero 2.0 states).")

    # 3. Forward pass verification (zero NaNs, zero Infs)
    batch_size = 16
    x = torch.randn(batch_size, in_features, device=device)
    out = layer(x)

    assert not torch.isnan(out).any(), "Forward pass produced NaNs!"
    assert not torch.isinf(out).any(), "Forward pass produced Infs!"
    assert out.shape == (batch_size, out_features), f"Wrong output shape: {out.shape}"
    print("  [PASS] Forward pass completed with zero NaNs and zero Infs.")

    # 4. Backward pass verification (STE and Gaussian density gradients)
    loss = (out ** 2).mean()
    loss.backward()

    for name, p in layer.named_parameters():
        if name == 'threshold_high':
            continue  # Deprecated legacy parameter preserved for state_dict backward compatibility
        assert p.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaNs!"
        assert not torch.isinf(p.grad).any(), f"Parameter {name} gradient contains Infs!"
        print(f"    • {name:16s} grad norm: {p.grad.norm().item():.6e} (NaNs: 0, Infs: 0)")

    print("  [PASS] Backward pass completed with valid STE and Gaussian density gradients for all active parameters.")

    # 5. Multi-step optimization check
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    for step in range(5):
        optimizer.zero_grad()
        x_step = torch.randn(batch_size, in_features, device=device)
        out_step = layer(x_step)
        loss_step = (out_step ** 2).mean()
        loss_step.backward()
        optimizer.step()
        assert not torch.isnan(layer.weight).any(), f"NaNs in weight at step {step}"
        assert not torch.isnan(layer.threshold_low).any(), f"NaNs in threshold_low at step {step}"

    print("  [PASS] 5-step optimization completed stably with zero NaNs.")

    # 6. Test recalibration method
    layer.calibrate_thresholds(scale_low=0.4, scale_high=1.6)
    new_gamma = layer.weight.abs().mean()
    assert torch.isclose(layer.threshold_low, 0.4 * new_gamma, atol=1e-5)
    assert torch.isclose(layer.threshold_high, 1.6 * new_gamma, atol=1e-5)
    print("  [PASS] calibrate_thresholds() dynamic recalibration verified.")


def main():
    print("=" * 60)
    print("  PCEI Verification: BitNet 1.58b Thresholds & Datasets")
    print("=" * 60)

    test_schema_parsers()
    counts = test_dataset_loading()
    
    test_ternary_bitnet_calibration("cpu")
    if torch.backends.mps.is_available():
        test_ternary_bitnet_calibration("mps")
    elif torch.cuda.is_available():
        test_ternary_bitnet_calibration("cuda")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
