"""
Evaluate the full detection + classification pipeline on a folder of images.
Mimics the iOS app workflow: YOLO detection -> crop -> classifier.

Accepts multiple checkpoints to compare models side by side (e.g. s1 vs s2).

Usage:
    uv run python -m scripts.evaluate_pipeline \\
        --images images/egypt/ \\
        --checkpoints checkpoints/s1/.../best.ckpt checkpoints/s2/.../best.ckpt \\
        --output results/egypt/

    # skip detection, classify full images directly
    uv run python -m scripts.evaluate_pipeline \\
        --images images/egypt/ \\
        --checkpoints checkpoints/s2/.../best.ckpt \\
        --no-detect --output results/egypt/
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.model import create_model
from src.transforms import val_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DETECTOR_PATH = "detector/yolov8m-worldv2.pt"
DETECTOR_IMGSZ = 1280
DETECTOR_CONF = 0.15

# Box colors per model index
_COLORS = ["#00FF7F", "#FF6B6B", "#4FC3F7", "#FFD54F"]


def load_label_names(cache_dir="data/hf_cache") -> list[str]:
    from datasets import load_from_disk

    ds = load_from_disk(cache_dir)
    return ds.features["label"].names


def load_classifier(ckpt_path: Path):
    """Load a classifier from a checkpoint. Returns (model, image_size, label)."""
    run_dir = ckpt_path.parent

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    backbone = metrics["backbone"]
    num_classes = config["dataset"]["num_classes"]
    image_size = config["data"]["image_size"]

    model = create_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    parts = run_dir.parts
    session = parts[-2] if len(parts) >= 2 else "?"
    label = f"{session} / {backbone}"

    return model, image_size, label


def classify_crops(
    model, crops: list[Image.Image], image_size: int, top_k: int
) -> list[list[tuple]]:
    """Classify a list of PIL crops. Returns list of [(prob, idx), ...] per crop."""
    transform = val_transforms(image_size)
    tensors = torch.stack([transform(c) for c in crops])
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1)
    results = []
    for row in probs:
        top_probs, top_idx = row.topk(top_k)
        results.append(list(zip(top_probs.tolist(), top_idx.tolist())))
    return results


def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


CANVAS_W = 500
PADDING = 16
LINE_H = 24


def save_crop_with_labels(
    crop: Image.Image,
    det: dict,
    model_labels: list[str],
    output_path: Path,
) -> None:
    """White canvas: crop scaled and centered on top, label text below."""
    font_label = _load_font(15)
    font_conf = _load_font(13)

    # scale crop to fit CANVAS_W while keeping aspect ratio
    scale = CANVAS_W / crop.width
    img_w = CANVAS_W
    img_h = int(crop.height * scale)
    scaled = crop.resize((img_w, img_h), Image.Resampling.LANCZOS)

    text_area_h = PADDING + len(model_labels) * LINE_H + PADDING
    canvas_h = img_h + text_area_h
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), color=(255, 255, 255))
    canvas.paste(scaled, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # subtle separator line between image and text
    draw.line([(0, img_h), (CANVAS_W, img_h)], fill=(220, 220, 220), width=1)

    for i, model_label in enumerate(model_labels):
        preds = det["predictions"][model_label]
        species = preds[0]["label"].replace("_", " ")
        prob = preds[0]["prob"]
        color = _COLORS[i % len(_COLORS)]

        ty = img_h + PADDING + i * LINE_H
        draw.text(
            (PADDING, ty), f"[{model_label}]", fill=(100, 100, 100), font=font_conf
        )

        label_x = PADDING + 160
        draw.text((label_x, ty), species, fill=(30, 30, 30), font=font_label)

        prob_text = f"{prob:.0%}"
        prob_x = CANVAS_W - PADDING - draw.textlength(prob_text, font=font_label)
        draw.text((prob_x, ty), prob_text, fill=color, font=font_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO + classifier pipeline")
    parser.add_argument(
        "--images", type=str, required=True, help="Image file or directory"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more .ckpt paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save annotated images and results.json",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top-k predictions per box"
    )
    parser.add_argument(
        "--conf", type=float, default=DETECTOR_CONF, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--no-detect", action="store_true", help="Skip YOLO, classify full image"
    )
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading label names from cache...")
    label_names = load_label_names()

    print("Loading classifiers...")
    classifiers = [load_classifier(Path(c)) for c in args.checkpoints]
    for _, _, lbl in classifiers:
        print(f"  loaded: {lbl}")
    model_labels = [lbl for _, _, lbl in classifiers]

    if not args.no_detect:
        print(f"Loading detector: {DETECTOR_PATH}")
        detector = YOLO(DETECTOR_PATH)
        detector.set_classes(["fish"])

    image_path = Path(args.images)
    if image_path.is_dir():
        files = sorted(
            p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        files = [image_path]

    print(f"\nProcessing {len(files)} image(s)...\n")
    sep = "─" * 70

    all_results: dict = {}

    for img_path in files:
        print(sep)
        print(f"  {img_path.name}")
        img = Image.open(img_path).convert("RGB")

        if args.no_detect:
            raw_boxes = [(0, 0, img.width, img.height)]
            raw_confs = [1.0]
            crops = [img]
        else:
            det_result = detector(
                img, imgsz=DETECTOR_IMGSZ, conf=args.conf, verbose=False
            )[0]
            boxes = det_result.boxes
            if len(boxes) == 0:
                print("  no fish detected\n")
                all_results[img_path.name] = []
                continue
            raw_boxes, raw_confs, crops = [], [], []
            for xyxy, conf in zip(boxes.xyxy, boxes.conf):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                raw_boxes.append((x1, y1, x2, y2))
                raw_confs.append(float(conf))
                crops.append(img.crop((x1, y1, x2, y2)))

        # classify all crops for each model
        model_preds = {}
        for model, image_size, model_label in classifiers:
            model_preds[model_label] = classify_crops(
                model, crops, image_size, args.top_k
            )

        # build detection records
        detections = []
        for i, (box, yolo_conf) in enumerate(zip(raw_boxes, raw_confs)):
            det: dict = {
                "box": list(box),
                "yolo_conf": round(yolo_conf, 4),
                "predictions": {},
            }
            for model_label in model_labels:
                det["predictions"][model_label] = [
                    {"label": label_names[idx], "prob": round(prob, 4)}
                    for prob, idx in model_preds[model_label][i]
                ]
            detections.append(det)

        all_results[img_path.name] = detections

        # print to terminal
        for i, det in enumerate(detections):
            print(f"\n  box {i + 1} (YOLO conf {det['yolo_conf']:.2f})")
            for model_label in model_labels:
                top_str = "  │  ".join(
                    f"{p['label']} {p['prob']:.1%}"
                    for p in det["predictions"][model_label]
                )
                print(f"    [{model_label}]  {top_str}")

        # save one crop per detection with labels drawn in
        if out_dir:
            for i, (det, crop) in enumerate(zip(detections, crops)):
                out_name = f"{img_path.stem}_box{i + 1}{img_path.suffix}"
                save_crop_with_labels(crop, det, model_labels, out_dir / out_name)

    print(sep)

    # save results.json
    if out_dir:
        results_path = out_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved annotated images and results.json to {out_dir}/")


if __name__ == "__main__":
    main()
