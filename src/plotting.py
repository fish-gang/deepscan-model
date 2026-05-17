import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.model import create_model

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_GREEN = "#2ca02c"
_BLUE = "#1f77b4"

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


def _load_metrics(run_path: Path) -> dict:
    path = run_path / "metrics.json" if run_path.is_dir() else run_path
    with open(path) as f:
        return json.load(f)


def _run_labels(runs: list[dict], paths: list[Path]) -> list[str]:
    backbones = [r.get("backbone", f"run_{i}") for i, r in enumerate(runs)]
    counts = Counter(backbones)
    labels, seen = [], Counter()
    for backbone in backbones:
        if counts[backbone] > 1:
            seen[backbone] += 1
            labels.append(f"{backbone.replace('_', ' ')} ({seen[backbone]})")
        else:
            labels.append(backbone.replace("_", " "))
    return labels


def _param_count_m(backbone: str) -> float:
    model = create_model(num_classes=1, backbone=backbone, pretrained=False)
    return sum(p.numel() for p in model.parameters()) / 1e6


def _save_or_show(fig, output: str | None) -> None:
    plt.tight_layout()
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_curves(run_paths: list[str], output: str | None = None) -> None:
    paths = [Path(p) for p in run_paths]
    runs = [_load_metrics(p) for p in paths]
    labels = _run_labels(runs, paths)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DeepScan — Training History", fontsize=15, fontweight="bold", y=1.01)

    for i, (run, label) in enumerate(zip(runs, labels)):
        color = _PALETTE[i % len(_PALETTE)]
        epochs = run["epochs"]

        if run.get("train_loss"):
            ax_loss.plot(epochs, run["train_loss"], color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} — train")
            ax_loss.fill_between(epochs, run["train_loss"], run["val_loss"], color=color, alpha=0.07)
        ax_loss.plot(epochs, run["val_loss"], color=color, linestyle="-", linewidth=2.2, label=f"{label} — val")

        if run.get("train_acc"):
            ax_acc.plot(epochs, run["train_acc"], color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} — train")
            ax_acc.fill_between(epochs, run["train_acc"], run["val_acc"], color=color, alpha=0.07)
        ax_acc.plot(epochs, run["val_acc"], color=color, linestyle="-", linewidth=2.2, label=f"{label} — val")

        best_idx = run["val_acc"].index(max(run["val_acc"]))
        ax_acc.scatter(epochs[best_idx], run["val_acc"][best_idx], color=color, s=70, zorder=5, marker="*")

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

    _save_or_show(fig, output)


def plot_model_comparison(run_paths: list[str], output: str | None = None) -> None:
    paths = [Path(p) for p in run_paths]
    runs = [_load_metrics(p) for p in paths]

    entries = [
        (r["backbone"], r["test_acc"], max(r["val_acc"]))
        for r in runs
        if "test_acc" in r
    ]
    if not entries:
        print("No test_acc found in any run, skipping summary plot.")
        return

    entries.sort(key=lambda x: x[1])  # ascending — best ends up at top
    backbones, test_accs, val_accs = zip(*entries)

    params = [_param_count_m(b) for b in backbones]
    best_idx = len(test_accs) - 1  # last after ascending sort
    colors = [_GREEN if i == best_idx else _BLUE for i in range(len(backbones))]
    labels = [b.replace("_", " ") for b in backbones]

    fig_height = max(4.0, len(backbones) * 0.9 + 1.5)
    fig, (ax_acc, ax_params) = plt.subplots(
        1, 2, figsize=(12, fig_height), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle("DeepScan — Model Comparison", fontsize=15, fontweight="bold")

    y = list(range(len(backbones)))

    # Accuracy panel
    bars = ax_acc.barh(y, [a * 100 for a in test_accs], color=colors, alpha=0.85, height=0.55)
    ax_acc.scatter(
        [a * 100 for a in val_accs], y,
        color=colors, marker="|", s=250, linewidths=2.5, zorder=5,
        label="Best val accuracy",
    )
    for bar, acc in zip(bars, test_accs):
        ax_acc.text(
            bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}", va="center", ha="right", fontsize=10,
            color="white", fontweight="bold",
        )
    ax_acc.annotate(
        "★ best",
        xy=(test_accs[best_idx] * 100, best_idx),
        xytext=(test_accs[best_idx] * 100 + 0.4, best_idx + 0.3),
        fontsize=9, color=_GREEN, fontweight="bold",
    )
    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(labels)
    ax_acc.set_xlabel("Accuracy (%)")
    ax_acc.set_title("Test Accuracy")
    x_min = min(test_accs) * 100 - 5
    ax_acc.set_xlim(left=max(0.0, x_min), right=102)
    ax_acc.legend(loc="lower right")

    # Params panel
    ax_params.barh(y, params, color=colors, alpha=0.85, height=0.55)
    for i, p in enumerate(params):
        ax_params.text(
            p + max(params) * 0.02, i,
            f"{p:.1f}M", va="center", ha="left", fontsize=10,
        )
    ax_params.set_yticks(y)
    ax_params.set_yticklabels([])
    ax_params.set_xlabel("Parameters (M)")
    ax_params.set_title("Model Size")
    ax_params.set_xlim(right=max(params) * 1.4)

    _save_or_show(fig, output)
