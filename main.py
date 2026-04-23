import argparse
import yaml
from types import SimpleNamespace
from src.trainer import train


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })


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

    train(config, config_path=args.config)


if __name__ == "__main__":
    main()