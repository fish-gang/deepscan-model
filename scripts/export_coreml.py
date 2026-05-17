"""
Export a trained DeepScan checkpoint to Core ML format.

Usage:
    uv run python -m scripts.export_coreml --checkpoint checkpoints/.../best.ckpt
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.export import export


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })


def main():
    parser = argparse.ArgumentParser(description="Export model to Core ML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    with open(ckpt_path.parent / "config.yaml") as f:
        config = dict_to_namespace(yaml.safe_load(f))

    export(ckpt_path.parent, config)


if __name__ == "__main__":
    main()
