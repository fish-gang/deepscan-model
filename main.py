import argparse
from pathlib import Path
import yaml
from types import SimpleNamespace

from src.evaluate import evaluate_metrics
from src.export import export_best
from src.plotting import plot_model_comparison, plot_training_curves
from src.trainer import train


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def main():
    parser = argparse.ArgumentParser(description="DeepScan fish classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--no-export", action="store_true", help="Skip Core ML export")
    args = parser.parse_args()

    with open(args.config) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    session = getattr(config, "session", "default")
    revision = config.dataset.revision.replace("/", "-")

    run_dirs = []
    for backbone in config.model.backbones:
        config.model.backbone = backbone
        run_dir = train(config, config_path=args.config)
        run_dirs.append(run_dir)
        # Create evaluation plots (confusion matrix + F1 scores) for this model
        evaluate_metrics(
            run_dir,
            config,
            Path("plots") / f"{session}_{revision}_{backbone}",
        )

    run_dir_strs = [str(r) for r in run_dirs]

    plot_training_curves(
        run_dir_strs, output=f"plots/{session}_{revision}_comparison.png"
    )
    plot_model_comparison(
        run_dir_strs, output=f"plots/{session}_{revision}_summary.png"
    )

    if not args.no_export:
        export_best(run_dirs, config)


if __name__ == "__main__":
    main()
