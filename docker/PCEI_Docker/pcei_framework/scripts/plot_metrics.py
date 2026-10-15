"""
PCEI Graphing Tools

Reads the JSON metric logs and generates matplotlib plots for:
1. Loss curves (total + per-modality)
2. Ternary weight distributions over time (grokking indicator)
3. Topology migration scatter plots
4. Confidence consolidation curves
5. Entanglement bond strength over time
6. Dream strength annealing curve
7. Gradient norm heatmaps

Usage:
    python scripts/plot_metrics.py <log_dir>
    
All plots saved as PNG in the log directory.
"""
import json
import sys
from pathlib import Path
from typing import Union, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Run: pip install matplotlib")


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_loss_curves(log_dir: Path):
    """Plot training loss over time — the primary grokking diagnostic."""
    data = load_json(log_dir / "loss_history.json")
    if not data:
        print("No loss data found.")
        return
    
    steps = [r["step"] for r in data]
    total = [r["loss_total"] for r in data]
    text = [r["loss_text"] for r in data]
    img = [r["loss_img"] for r in data]
    audio = [r["loss_audio"] for r in data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Total loss
    if len(steps) > 1:
        ax1.plot(steps, total, 'b-', linewidth=1.5, label='Total Loss', alpha=0.8)
        ax2.plot(steps, text, 'g-', linewidth=1, label='Text', alpha=0.7)
        ax2.plot(steps, img, 'r-', linewidth=1, label='Image', alpha=0.7)
        ax2.plot(steps, audio, 'm-', linewidth=1, label='Audio', alpha=0.7)
    else:
        ax1.scatter(steps, total, color='blue', label='Total Loss')
        ax2.scatter(steps, text, color='green', label='Text')
        ax2.scatter(steps, img, color='red', label='Image')
        ax2.scatter(steps, audio, color='magenta', label='Audio')
        
    ax1.set_ylabel('Loss')
    ax1.set_title('PCEI Training Loss — Grokking Watch', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Per-Modality Loss Breakdown', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "loss_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: loss_curves.png")


def plot_ternary_distributions(log_dir: Path):
    """Plot the evolution of {0,1,2} weight ratios — key grokking indicator.
    When grokking occurs, you see a sudden shift: many weights snap from 1 → 0 or 2."""
    data = load_json(log_dir / "ternary_distributions.json")
    if not data or len(data) == 0:
        print("No ternary distribution data found.")
        return
    
    # Aggregate across all layers at each step
    steps = []
    pct_0, pct_1, pct_2 = [], [], []
    
    for snapshot in data:
        steps.append(snapshot["step"])
        total_0, total_1, total_2, total_n = 0, 0, 0, 0
        for layer_name, dist in snapshot["layers"].items():
            n = dist["total_params"]
            total_0 += dist["pruned_0"] * n
            total_1 += dist["partial_1"] * n
            total_2 += dist["strong_2"] * n
            total_n += n
        if total_n > 0:
            pct_0.append(total_0 / total_n)
            pct_1.append(total_1 / total_n)
            pct_2.append(total_2 / total_n)
        else:
            pct_0.append(0); pct_1.append(0); pct_2.append(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    if len(steps) > 1:
        ax.stackplot(steps, pct_0, pct_1, pct_2, 
                     labels=['0 (Pruned)', '1 (Partial)', '2 (Strong)'],
                     colors=['#ff6b6b', '#ffd93d', '#6bcb77'], alpha=0.8)
    else:
        # Fallback for single-point Step 0
        ax.bar(steps, pct_0, color='#ff6b6b', label='0 (Pruned)', width=0.1)
        ax.bar(steps, pct_1, bottom=pct_0, color='#ffd93d', label='1 (Partial)', width=0.1)
        ax.bar(steps, pct_2, bottom=[a+b for a,b in zip(pct_0, pct_1)], color='#6bcb77', label='2 (Strong)', width=0.1)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction of Weights')
    ax.set_title('Metabolic Weight Distribution — Grokking = Sudden Shift', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "ternary_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: ternary_distributions.png")


def plot_topology_migration(log_dir: Path):
    """Scatter plot of PlasticLayer positions over training.
    Shows how entangled layers physically converge."""
    data = load_json(log_dir / "topology_snapshots.json")
    if not data or len(data) == 0:
        print("No topology data found.")
        return
    
    num_plots = min(3, len(data))
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)
    
    # Show first, middle, and last snapshots
    indices = [0, len(data)//2, len(data)-1][:len(data)]
    
    for ax, idx in zip(axes, indices):
        snapshot = data[idx]
        positions = snapshot["positions"]
        
        xs, ys, colors, labels = [], [], [], []
        for layer_id, pos in positions.items():
            xs.append(pos["x"])
            ys.append(pos["y"])
            colors.append(pos["z"])
            labels.append(layer_id)
        
        scatter = ax.scatter(xs, ys, c=colors, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=6, ha='center', va='bottom')
        ax.set_title(f'Step {snapshot["step"]}', fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('Topology Migration — Layer Positions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(log_dir / "topology_migration.png", dpi=150)
    plt.close()
    print(f"  Saved: topology_migration.png")


def plot_confidence(log_dir: Path):
    """Shows slow_field_confidence calcification over time per layer."""
    data = load_json(log_dir / "confidence_history.json")
    if not data or len(data) == 0:
        print("No confidence data found.")
        return
    
    # Collect per-layer time series
    layer_series = {}
    steps = []
    for snapshot in data:
        steps.append(snapshot["step"])
        for layer_id, info in snapshot["layers"].items():
            if layer_id not in layer_series:
                layer_series[layer_id] = []
            layer_series[layer_id].append(info["confidence"])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for layer_id, vals in layer_series.items():
        ax.plot(steps[:len(vals)], vals, linewidth=1.2, label=layer_id, alpha=0.7)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('slow_field_confidence')
    ax.set_title('Memory Calcification — Consolidation Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "confidence_calcification.png", dpi=150)
    plt.close()
    print(f"  Saved: confidence_calcification.png")


def plot_bonds(log_dir: Path):
    """Entanglement bond count and mean strength over time."""
    data = load_json(log_dir / "bond_history.json")
    if not data or len(data) == 0:
        print("No bond data found.")
        return
    
    steps = [r["step"] for r in data]
    counts = [r["bond_count"] for r in data]
    strengths = [r["mean_strength"] for r in data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1.plot(steps, counts, 'b-', linewidth=1.5)
    ax1.set_ylabel('Active Bond Count')
    ax1.set_title('Entanglement Bonds Over Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, strengths, 'r-', linewidth=1.5)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Mean Bond Strength')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "entanglement_bonds.png", dpi=150)
    plt.close()
    print(f"  Saved: entanglement_bonds.png")


def plot_dream_strength(log_dir: Path):
    """Dream strength annealing curve."""
    data = load_json(log_dir / "dream_strength.json")
    if not data or not data.get("dream_strength"):
        print("No dream strength data found.")
        return
    
    vals = data["dream_strength"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(vals)), vals, 'purple', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('dream_strength')
    ax.set_title('DreamSeeder Strength Annealing', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "dream_strength.png", dpi=150)
    plt.close()
    print(f"  Saved: dream_strength.png")


def generate_all_plots(log_dir: Union[str, Path]):
    """Generate all diagnostic plots from a metrics log directory."""
    if not HAS_MPL:
        print("Cannot generate plots: matplotlib not installed")
        return
    
    log_dir = Path(log_dir)
    print(f"\n[PCEI Plotter] Generating plots from: {log_dir}\n")
    
    plot_loss_curves(log_dir)
    plot_ternary_distributions(log_dir)
    plot_topology_migration(log_dir)
    plot_confidence(log_dir)
    plot_bonds(log_dir)
    plot_dream_strength(log_dir)
    
    print(f"\nAll plots saved to {log_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_metrics.py <log_dir>")
        sys.exit(1)
    generate_all_plots(sys.argv[1])
