"""
Exports the YOLO-World fish detector to CoreML for the iOS app.

Place a YOLO-World .pt file at detector/yolov8m-worldv2.pt before running.
Supports both predict (sanity-check) and export modes.

Usage:
    uv run python -m scripts.export_detector --predict
    uv run python -m scripts.export_detector --export
    uv run python -m scripts.export_detector --predict --export   # both
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLOWorld

# YOLO-World variant. "s" = small (~30MB CoreML, balanced), "m" = bigger/slower,
# "l" = largest. Bumped to "m" after first-pass results showed missed fish.
WEIGHTS_PATH = Path("detector/yolov8m-worldv2.pt")

# Single class - species identification is handled by the classifier.
CLASSES = ["fish"]

# Detection confidence cutoff during prediction sanity check.
CONF = 0.15

# Input image size at inference. 1280 gives the model 4× more pixels for
# finding small/distant fish in cluttered reef scenes.
IMGSZ = 1280


def export_detector(weights_path: Path = WEIGHTS_PATH) -> Path:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No detector weights found at {weights_path}. "
            "Download a YOLO-World .pt file and place it there."
        )

    print(f"Loading {weights_path}...")
    model = YOLOWorld(str(weights_path))
    model.set_classes(CLASSES)

    return _export(model)


def _export(model: YOLOWorld) -> Path:
    print("Exporting detector to CoreML... (this can take a minute)")
    exported = model.export(format="coreml", nms=True, imgsz=IMGSZ)

    output_path = Path("model") / "FishDetector.mlpackage"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.move(str(exported), str(output_path))

    print(f"Exported detector → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Run inference on test_photos/ and save annotated JPGs")
    parser.add_argument("--export", action="store_true",
                        help="Export the model to CoreML (model/FishDetector.mlpackage)")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH),
                        help=f"Path to YOLO-World .pt file (default: {WEIGHTS_PATH})")
    args = parser.parse_args()

    if not (args.predict or args.export):
        parser.print_help()
        return

    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"No detector weights at {weights_path}. "
              "Download a YOLO-World .pt and place it there.")
        return

    print(f"Loading {weights_path}...")
    model = YOLOWorld(str(weights_path))
    model.set_classes(CLASSES)

    if args.predict:
        photos_dir = Path("test_photos")
        if not photos_dir.exists() or not any(photos_dir.iterdir()):
            print(f"Put a few snorkel photos in {photos_dir}/ first, then re-run.")
            return
        print(f"Running detection on {photos_dir}/ at conf={CONF}...")
        model.predict(str(photos_dir), save=True, conf=CONF, imgsz=IMGSZ)
        print("Done. Open runs/detect/predict*/ to review the boxes.")

    if args.export:
        _export(model)


if __name__ == "__main__":
    main()
