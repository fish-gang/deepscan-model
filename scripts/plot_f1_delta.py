"""
Compare per-class F1-score between two checkpoints (e.g. s1 vs s2).
Shows delta (s2 - s1) as green/red bars.

Usage:
    uv run python -m scripts.plot_f1_delta \\
        --s1 checkpoints/s1/.../best.ckpt \\
        --s2 checkpoints/s2/.../best.ckpt \\
        --output plots/f1_delta.png
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from src.data import DeepScanDataModule
from src.model import create_model

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
    "font.size": 10,
})

GREEN = "#2ca02c"
RED   = "#d62728"


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
    model = create_model(num_classes=config.dataset.num_classes, backbone=backbone, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, backbone


def get_f1_scores(model, data: DeepScanDataModule, label_names: list[str]) -> dict[str, float]:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data.test_dataloader():
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    report = classification_report(
        np.array(all_labels), np.array(all_preds),
        target_names=label_names, output_dict=True
    )
    return {name: report[name]["f1-score"] for name in label_names if name in report}


def main():
    parser = argparse.ArgumentParser(description="F1-score delta plot between two checkpoints")
    parser.add_argument("--s1", type=str, required=True, help="Baseline checkpoint (s1)")
    parser.add_argument("--s2", type=str, required=True, help="New checkpoint (s2)")
    parser.add_argument("--output", type=str, default="plots/f1_delta.png")
    args = parser.parse_args()

    print("Loading s1 model...")
    model_s1, config, backbone_s1 = load_model_and_config(Path(args.s1))
    print("Loading s2 model...")
    model_s2, _, backbone_s2 = load_model_and_config(Path(args.s2))

    print("Loading test set...")
    data = DeepScanDataModule(config)
    data.setup("test")
    label_names = data.label_names

    print("Running inference...")
    f1_s1 = get_f1_scores(model_s1, data, label_names)
    f1_s2 = get_f1_scores(model_s2, data, label_names)

    # compute deltas, sort by delta ascending
    deltas = {name: f1_s2[name] - f1_s1[name] for name in label_names if name in f1_s1 and name in f1_s2}
    sorted_items = sorted(deltas.items(), key=lambda x: x[1])
    names  = [name.replace("_", " ") for name, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [GREEN if v >= 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.52)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85, height=0.6)

    for bar, val in zip(bars, values):
        sign = "+" if val >= 0 else ""
        x = bar.get_width()
        ha = "left" if val >= 0 else "right"
        offset = 0.002 if val >= 0 else -0.002
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}", va="center", ha=ha, fontsize=9, fontweight="bold",
                color=GREEN if val >= 0 else RED)

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1-Score Δ (s2 − s1)")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)

    session_s1 = Path(args.s1).parent.parts[-2]
    session_s2 = Path(args.s2).parent.parts[-2]
    fig.text(0.5, -0.02,
             f"{session_s1} ({backbone_s1})  →  {session_s2} ({backbone_s2})",
             ha="center", fontsize=9, color="#555555", style="italic")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
