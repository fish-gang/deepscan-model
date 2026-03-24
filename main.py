import argparse
import yaml

from src.trainer import train


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
        config = yaml.safe_load(f)

    train(config, config_path=args.config)


if __name__ == "__main__":
    main()
