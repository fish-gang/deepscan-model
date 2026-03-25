"""
Run inference on new images using a trained DeepScan checkpoint.

Usage:
    uv run python -m scripts.predict --checkpoint checkpoints/.../best.ckpt --image fish.jpg
    uv run python -m scripts.predict --checkpoint checkpoints/.../best.ckpt --image images/
"""

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.data import load_deepscan_dataset
from src.model import create_model
from src.transforms import val_transforms


def load_model(checkpoint_path: str):
    """Load a trained model from a checkpoint directory."""
    ckpt_path = Path(checkpoint_path)
    config_path = ckpt_path.parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
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

    image_size = config["data"]["image_size"]
    return model, image_size


def predict_image(
    model, image_path: str, image_size: int, label_names: list[str], top_k: int = 3
):
    """Run prediction on a single image."""
    image = Image.open(image_path).convert("RGB")
    transform = val_transforms(image_size)
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)

    print(f"\n{Path(image_path).name}:")
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {label_names[idx]:30s} {prob:.4%}")


def main():
    parser = argparse.ArgumentParser(description="Predict fish species from images")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to image or folder of images"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top predictions to show"
    )
    args = parser.parse_args()

    dataset = load_deepscan_dataset()
    label_names = dataset.features["label"].names

    model, image_size = load_model(args.checkpoint)

    image_path = Path(args.image)
    if image_path.is_dir():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []
        for ext in extensions:
            files.extend(image_path.glob(ext))
        for img_file in sorted(files):
            predict_image(model, str(img_file), image_size, label_names, args.top_k)
    else:
        predict_image(model, args.image, image_size, label_names, args.top_k)


if __name__ == "__main__":
    main()
