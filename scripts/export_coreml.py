"""
Export a trained DeepScan checkpoint to Core ML format.

Usage:
    uv run python export_coreml.py --checkpoint checkpoints/.../best.ckpt
"""

import argparse
from pathlib import Path

import torch
import yaml
import coremltools as ct

from src.data import load_deepscan_dataset
from src.model import create_model


def export(checkpoint_path: str):
    ckpt_path = Path(checkpoint_path)
    config_path = ckpt_path.parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    image_size = config["data"]["image_size"]

    # Load model
    model = create_model(
        num_classes=model_cfg["num_classes"],
        backbone=model_cfg["backbone"],
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    # Get label names from dataset
    dataset = load_deepscan_dataset()
    label_names = dataset.features["label"].names

    # Trace the model
    example_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1 / 255.0,
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            )
        ],
        classifier_config=ct.ClassifierConfig(class_labels=label_names),
        convert_to="mlprogram",
    )

    # Save
    output_path = ckpt_path.parent / f"DeepScan_{model_cfg['backbone']}.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model to Core ML")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    args = parser.parse_args()
    export(args.checkpoint)


if __name__ == "__main__":
    main()
