"""
Plot training history for one or more DeepScan runs.

Usage:
    uv run python -m scripts.plot_training checkpoints/run1
    uv run python -m scripts.plot_training checkpoints/run1 checkpoints/run2 --output plots/comparison.png
"""

import argparse

from src.plotting import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Plot DeepScan training history")
    parser.add_argument("runs", nargs="+", help="Run directories (or metrics.json paths)")
    parser.add_argument("--output", "-o", help="Output path. Displays interactively if omitted.")
    args = parser.parse_args()
    plot_training_curves(args.runs, args.output)


if __name__ == "__main__":
    main()
