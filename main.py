import argparse
import yaml
from types import SimpleNamespace

from scripts.plot_training import plot
from src.trainer import train


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def main():
    parser = argparse.ArgumentParser(description="DeepScan fish classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    run_dirs = []
    for backbone in config.model.backbones:
        config.model.backbone = backbone
        run_dir = train(config, config_path=args.config)
        run_dirs.append(str(run_dir))

    revision = config.dataset.revision.replace("/", "-")
    output = f"plots/{revision}_comparison.png"
    plot(run_dirs, output=output)
    print(f"Saved comparison plot to {output}")


if __name__ == "__main__":
    main()
