"""
Compute confusion matrix and per-class F1-score on the test set.

Usage:
    uv run python -m scripts.evaluate_metrics \\
        --checkpoint checkpoints/s2/.../best.ckpt \\
        --output plots/s2_v1.0_efficientnet_b0/
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.evaluate import evaluate_metrics


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix and F1-score on test set"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best.ckpt"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent

    with open(run_dir / "config.yaml") as f:
        config = dict_to_namespace(yaml.safe_load(f))

    evaluate_metrics(run_dir, config, Path(args.output))


if __name__ == "__main__":
    main()
