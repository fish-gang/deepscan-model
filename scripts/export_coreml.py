"""
Export a trained DeepScan checkpoint to Core ML format.

Usage:
    uv run python export_coreml.py --checkpoint checkpoints/.../best.ckpt
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import coremltools as ct

from src.data import load_deepscan_dataset
from src.model import create_model


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })


def export(checkpoint_path: str):
    ckpt_path = Path(checkpoint_path)
    config_path = ckpt_path.parent / "config.yaml"

    with open(config_path) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    # Load model
    model = create_model(
        num_classes=config.dataset.num_classes,
        backbone=config.model.backbone,
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
    dataset = load_deepscan_dataset(
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        cache_dir=Path(config.dataset.cache_dir),
    )
    label_names = dataset.features["label"].names

    # Trace the model
    image_size = config.data.image_size
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

    # Metadata
    mlmodel.short_description = (
        f"Tropical reef fish classifier — {config.dataset.num_classes} classes, "
        f"{config.model.backbone} backbone, trained on {config.dataset.repo_id} {config.dataset.revision}"
    )
    mlmodel.input_description["image"] = (
        f"RGB image ({image_size}x{image_size}), normalised with "
        f"ImageNet mean/std"
    )
    mlmodel.output_description["classLabel"] = (
        "Predicted species label (scientific name) or 'unknown_fish' / 'no_fish'"
    )
    mlmodel.output_description["classLabel_probs"] = (
        "Class probabilities for all 14 classes"
    )

    mlmodel.user_defined_metadata["backbone"] = config.model.backbone
    mlmodel.user_defined_metadata["num_classes"] = str(config.dataset.num_classes)
    mlmodel.user_defined_metadata["image_size"] = str(image_size)
    mlmodel.user_defined_metadata["dataset_revision"] = config.dataset.revision
    mlmodel.user_defined_metadata["checkpoint"] = ckpt_path.name

    # Save
    output_path = Path("model") / "DeepScanClassifier.mlpackage"

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
