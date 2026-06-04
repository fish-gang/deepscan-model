"""Per-checkpoint test-set evaluation: confusion matrix and per-class F1-score."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data import DeepScanDataModule
from src.model import create_model

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


def _plot_confusion_matrix(
    cm: np.ndarray, label_names: list[str], output_path: Path, model_label: str
) -> None:
    n = len(label_names)
    fig_size = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)

    short_names = [name.replace("_", "\n") for name in label_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.grid(False)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count == 0:
                continue
            pct = cm_norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center", fontsize=7, color=color)

    fig.text(0.5, -0.01, model_label, ha="center", fontsize=9, color="#555555", style="italic")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def _plot_f1_scores(
    report: dict, label_names: list[str], output_path: Path, model_label: str
) -> None:
    scores_and_names = sorted(
        (report[n]["f1-score"], n.replace("_", " ")) for n in label_names if n in report
    )
    f1_scores, names = zip(*scores_and_names)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), f1_scores, color="#1f77b4", alpha=0.85, height=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center", ha="right", fontsize=9, color="white", fontweight="bold",
        )

    macro_f1 = report["macro avg"]["f1-score"]
    ax.axvline(macro_f1, color="black", linestyle="--", linewidth=1.2, label=f"Macro F1: {macro_f1:.2f}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    fig.text(0.5, -0.02, model_label, ha="center", fontsize=9, color="#555555", style="italic")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved F1 plot to {output_path}")


def evaluate_metrics(run_dir: Path, config: SimpleNamespace, output_dir: Path) -> None:
    """Run test-set inference and save confusion_matrix.png + f1_scores.png to output_dir."""
    with open(run_dir / "metrics.json") as f:
        backbone = json.load(f)["backbone"]

    model = create_model(num_classes=config.dataset.num_classes, backbone=backbone, pretrained=False)
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    data = DeepScanDataModule(config)
    data.setup()
    label_names = data.label_names

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data.test_dataloader():
            all_preds.extend(model(images).argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=label_names))
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}  |  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    session = run_dir.parts[-2] if len(run_dir.parts) >= 2 else "?"
    model_label = f"{session}  ·  {backbone}  ·  Dataset {config.dataset.revision}"

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, label_names, output_dir / "confusion_matrix.png", model_label)
    _plot_f1_scores(report, label_names, output_dir / "f1_scores.png", model_label)
