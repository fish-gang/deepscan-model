"""
Plot training history for one or more DeepScan runs.

Usage:
    uv run python -m scripts.plot_training checkpoints/run1
    uv run python -m scripts.plot_training checkpoints/run1 checkpoints/run2 --output plots/comparison.png
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Tableau 10 palette
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.0,
})


def load_metrics(run_path: Path) -> dict:
    path = run_path / "metrics.json" if run_path.is_dir() else run_path
    with open(path) as f:
        return json.load(f)


def _run_labels(runs: list[dict], paths: list[Path]) -> list[str]:
    backbones = [r.get("backbone", f"run_{i}") for i, r in enumerate(runs)]
    counts = Counter(backbones)
    labels = []
    seen: Counter = Counter()
    for i, (backbone, path) in enumerate(zip(backbones, paths)):
        if counts[backbone] > 1:
            seen[backbone] += 1
            labels.append(f"{backbone.replace('_', ' ')} ({seen[backbone]})")
        else:
            labels.append(backbone.replace("_", " "))
    return labels


def plot(run_paths: list[str], output: str | None = None) -> None:
    paths = [Path(p) for p in run_paths]
    runs = [load_metrics(p) for p in paths]
    labels = _run_labels(runs, paths)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DeepScan — Training History", fontsize=15, fontweight="bold", y=1.01)

    for i, (run, label) in enumerate(zip(runs, labels)):
        color = _PALETTE[i % len(_PALETTE)]
        epochs = run["epochs"]

        # Loss curves
        if run.get("train_loss"):
            ax_loss.plot(epochs, run["train_loss"], color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} — train")
            ax_loss.fill_between(epochs, run["train_loss"], run["val_loss"], color=color, alpha=0.07)
        ax_loss.plot(epochs, run["val_loss"], color=color, linestyle="-", linewidth=2.2, label=f"{label} — val")

        # Accuracy curves
        if run.get("train_acc"):
            ax_acc.plot(epochs, run["train_acc"], color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} — train")
            ax_acc.fill_between(epochs, run["train_acc"], run["val_acc"], color=color, alpha=0.07)
        ax_acc.plot(epochs, run["val_acc"], color=color, linestyle="-", linewidth=2.2, label=f"{label} — val")

        # Mark best val accuracy epoch
        best_idx = run["val_acc"].index(max(run["val_acc"]))
        ax_acc.scatter(epochs[best_idx], run["val_acc"][best_idx], color=color, s=70, zorder=5, marker="*")

        # Dotted line for held-out test accuracy
        if "test_acc" in run:
            ax_acc.axhline(
                run["test_acc"], color=color, linestyle=":", linewidth=1.3, alpha=0.6,
                label=f"{label} — test {run['test_acc']:.1%}",
            )

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.legend(loc="upper right")

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    min_acc = min(min(r["val_acc"]) for r in runs)
    ax_acc.set_ylim(bottom=max(0.0, min_acc - 0.05))
    ax_acc.legend(loc="lower right")

    plt.tight_layout()

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DeepScan training history")
    parser.add_argument("runs", nargs="+", help="Run directories (or metrics.json paths)")
    parser.add_argument("--output", "-o", help="Output image path (e.g. plots/comparison.png). Displays interactively if omitted.")
    args = parser.parse_args()
    plot(args.runs, args.output)


if __name__ == "__main__":
    main()
