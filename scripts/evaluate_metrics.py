"""
Compute confusion matrix and per-class F1-score on the test set.

Usage:
    uv run python -m scripts.evaluate_metrics \\
        --checkpoint checkpoints/s2/.../best.ckpt \\
        --output results/metrics_s2/
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data import DeepScanDataModule
from src.model import create_model
from src.transforms import val_transforms


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


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def load_model_and_config(ckpt_path: Path):
    run_dir = ckpt_path.parent
    with open(run_dir / "config.yaml") as f:
        config = dict_to_namespace(yaml.safe_load(f))
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    backbone = metrics["backbone"]
    model = create_model(
        num_classes=config.dataset.num_classes, backbone=backbone, pretrained=False
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, backbone


def plot_confusion_matrix(
    cm: np.ndarray, label_names: list[str], output_path: Path, model_label: str = ""
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
    ax.set_xticklabels(
        short_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=8
    )
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.grid(False)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = cm_norm[i, j]
            if count == 0:
                continue
            color = "white" if pct > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n{pct:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    if model_label:
        fig.text(
            0.5,
            -0.01,
            model_label,
            ha="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def plot_f1_scores(
    report: dict, label_names: list[str], output_path: Path, model_label: str = ""
) -> None:
    f1_scores = [report[name]["f1-score"] for name in label_names if name in report]
    names = [name.replace("_", " ") for name in label_names if name in report]

    pairs = sorted(zip(f1_scores, names))
    f1_scores, names = zip(*pairs)

    colors = ["#1f77b4"] * len(f1_scores)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), f1_scores, color=colors, alpha=0.85, height=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            ha="right",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    macro_f1 = report["macro avg"]["f1-score"]
    ax.axvline(
        macro_f1,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Macro F1: {macro_f1:.2f}",
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    if model_label:
        fig.text(
            0.5,
            -0.02,
            model_label,
            ha="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved F1 plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix and F1-score on test set"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Directory for output files"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, config, backbone = load_model_and_config(ckpt_path)

    print("Loading test set...")
    data = DeepScanDataModule(config)
    data.setup("test")
    label_names = data.label_names

    print(f"Running inference on {len(data.test_ds)} test images...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data.test_dataloader():
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True
    )
    print(classification_report(all_labels, all_preds, target_names=label_names))

    session = ckpt_path.parent.parts[-2] if len(ckpt_path.parent.parts) >= 2 else "?"
    model_label = f"{session}  ·  {backbone}  ·  Dataset {config.dataset.revision}"

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm, label_names, out_dir / "confusion_matrix.png", model_label
    )
    plot_f1_scores(report, label_names, out_dir / "f1_scores.png", model_label)

    print(f"\nMacro F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
