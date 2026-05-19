"""
Exports a YOLO-World fish detector to CoreML for the iOS app.

Two-step workflow:

    1) Run with `--predict` to sanity-check detections on real snorkel
       photos in test_photos/. Inspect the saved JPGs in runs/detect/.

    2) Run with `--export` to produce FishDetector.mlpackage. Drag the
       resulting file into Xcode's DeepScan/ML/ folder.

Usage:
    uv run python scripts/export_detector.py --predict
    uv run python scripts/export_detector.py --export
    uv run python scripts/export_detector.py --predict --export   # both
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLOWorld

# YOLO-World variant. "s" = small (~30MB CoreML, balanced), "m" = bigger/slower,
# "l" = largest. Bumped to "m" after first-pass results showed missed fish.
BASE_WEIGHTS = "yolov8m-worldv2.pt"

# Single class — we don't care about species at the detector level. The
# classifier handles species. This keeps the detector generic and robust.
CLASSES = ["fish"]

# Detection confidence cutoff during prediction sanity check.
# Lowered from 0.25 to catch fish the model flagged with weaker confidence.
CONF = 0.15

# Input image size at inference. 640 is the YOLO-World default; 1280 gives
# the model 4× more pixels for finding small/distant fish in cluttered reef
# scenes — biggest single accuracy win, mildly slower at runtime.
IMGSZ = 1280


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Run inference on test_photos/ and save annotated JPGs")
    parser.add_argument("--export", action="store_true",
                        help="Export the model to CoreML (FishDetector.mlpackage)")
    args = parser.parse_args()

    if not (args.predict or args.export):
        parser.print_help()
        return

    print(f"Loading {BASE_WEIGHTS}...")
    model = YOLOWorld(BASE_WEIGHTS)
    model.set_classes(CLASSES)

    if args.predict:
        photos_dir = Path("test_photos")
        if not photos_dir.exists() or not any(photos_dir.iterdir()):
            print(f"⚠️  Put a few snorkel photos in {photos_dir}/ first, then re-run.")
            return
        print(f"Running detection on {photos_dir}/ at conf={CONF}...")
        model.predict(str(photos_dir), save=True, conf=CONF, imgsz=IMGSZ)
        print("✅ Done. Open runs/detect/predict*/ to review the boxes.")

    if args.export:
        print("Exporting to CoreML... (this can take a minute)")
        exported = model.export(format="coreml", nms=True, imgsz=IMGSZ)
        # Ultralytics returns the path of the exported package.
        src = Path(exported)
        dst = Path("FishDetector.mlpackage")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"✅ Exported: {dst.resolve()}")
        print("   Drag this folder into Xcode's DeepScan/ML/ group (Copy items if needed).")


if __name__ == "__main__":
    main()
