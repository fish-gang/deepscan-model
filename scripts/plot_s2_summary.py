"""Temporary script — s2 accuracy vs model size comparison plot."""

from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
    }
)

DATA = [
    ("SqueezeNet 1.1", 83.3, 1.2),
    ("ShuffleNet V2 x1.0", 88.0, 2.3),
    ("MobileNetV3 Small", 89.4, 2.5),
    ("MobileNetV3 Large", 93.6, 5.5),
    ("EfficientNet-B0", 94.0, 5.3),
    ("ResNet-50", 94.5, 25.0),
]

# sort ascending by accuracy so best ends up at top
DATA.sort(key=lambda x: x[1])
labels = [d[0] for d in DATA]
accs = [d[1] for d in DATA]
params = [d[2] for d in DATA]
y = list(range(len(DATA)))
BLUE = "#1f77b4"

fig, (ax_acc, ax_params) = plt.subplots(
    1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1]}
)

# --- accuracy panel ---
bars_acc = ax_acc.barh(y, accs, color=BLUE, alpha=0.85, height=0.55)
for bar, acc in zip(bars_acc, accs):
    ax_acc.text(
        bar.get_width() - 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{acc:.1f}%",
        va="center",
        ha="right",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
ax_acc.set_yticks(y)
ax_acc.set_yticklabels(labels, fontsize=9)
ax_acc.set_xlabel("Top-1 Accuracy (%)")
ax_acc.set_xlim(79, 101)
ax_acc.set_xticks(range(80, 101, 5))
ax_acc.yaxis.grid(False)
ax_acc.xaxis.grid(True)

# --- params panel ---
bars_p = ax_params.barh(y, params, color=BLUE, alpha=0.85, height=0.55)
for bar, p in zip(bars_p, params):
    ax_params.text(
        bar.get_width() + max(params) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{p:.1f}M",
        va="center",
        ha="left",
        fontsize=9,
    )
ax_params.set_yticks(y)
ax_params.set_yticklabels([])
ax_params.set_xlabel("Parameters (M)")
ax_params.set_xlim(right=max(params) * 1.35)
ax_params.yaxis.grid(False)
ax_params.xaxis.grid(True)

plt.tight_layout()
out = Path("plots/s2_model_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
