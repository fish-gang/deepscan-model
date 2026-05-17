import json
from pathlib import Path
from types import SimpleNamespace

import torch
import coremltools as ct

from src.data import load_deepscan_dataset
from src.model import create_model


def _resolve_backbone(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)["backbone"]
    return "_".join(run_dir.name.split("_")[2:])


def export(run_dir: Path, config: SimpleNamespace) -> Path:
    ckpt_path = run_dir / "best.ckpt"
    backbone = _resolve_backbone(run_dir)

    model = create_model(
        num_classes=config.dataset.num_classes,
        backbone=backbone,
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

    dataset = load_deepscan_dataset(
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        cache_dir=Path(config.dataset.cache_dir),
    )
    label_names = dataset.features["label"].names

    image_size = config.data.image_size
    example_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, example_input)

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

    mlmodel.short_description = (
        f"Tropical reef fish classifier — {config.dataset.num_classes} classes, "
        f"{backbone} backbone, trained on {config.dataset.repo_id} {config.dataset.revision}"
    )
    mlmodel.input_description["image"] = (
        f"RGB image ({image_size}x{image_size}), normalised with ImageNet mean/std"
    )
    mlmodel.output_description["classLabel"] = (
        "Predicted species label (scientific name) or 'unknown_fish' / 'no_fish'"
    )
    mlmodel.output_description["classLabel_probs"] = (
        "Class probabilities for all 14 classes"
    )
    mlmodel.user_defined_metadata["backbone"] = backbone
    mlmodel.user_defined_metadata["num_classes"] = str(config.dataset.num_classes)
    mlmodel.user_defined_metadata["image_size"] = str(image_size)
    mlmodel.user_defined_metadata["dataset_revision"] = config.dataset.revision
    mlmodel.user_defined_metadata["checkpoint"] = ckpt_path.name

    output_path = Path("model") / "DeepScanClassifier.mlpackage"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Exported {backbone} → {output_path}")
    return output_path


def export_best(run_dirs: list[Path], config: SimpleNamespace) -> Path | None:
    best_run, best_acc = None, -1.0
    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        acc = metrics.get("test_acc", -1.0)
        if acc > best_acc:
            best_acc = acc
            best_run = Path(run_dir)

    if best_run is None:
        print("No runs with metrics found, skipping export.")
        return None

    backbone = _resolve_backbone(best_run)
    print(f"Best model: {backbone} ({best_acc:.1%}) — exporting to Core ML...")
    return export(best_run, config)
