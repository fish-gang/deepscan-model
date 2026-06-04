# DeepScan Classification Model

Image classification model for tropical reef fish species. Trains multiple backbone architectures, auto-generates comparison plots, and exports the best model to Core ML for on-device iOS inference.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/fish-gang/deepscan-model.git
cd deepscan-model
uv sync
```

`uv sync` automatically creates a virtual environment with the correct Python version.

## Configuration

Training is configured via YAML files under `configs/`. The default config is `configs/default.yaml`. Key fields:

```yaml
session: "s2" # groups checkpoints and plots — increment per new run

dataset:
  repo_id: "fish-gang/deepscan-dataset"
  revision: "v1.0"
  num_classes: 14
  force_download: false # set to true to re-download the dataset

model:
  backbones: # all listed backbones are trained sequentially
    - squeezenet1_1
    - mobilenet_v3_small
    - shufflenet_v2_x1_0
    - mobilenet_v3_large
    - efficientnet_b0
    - resnet50
  pretrained: true

training:
  max_epochs: 40
  lr: 0.0001
```

## Training

```bash
uv run python main.py
```

With a custom config:

```bash
uv run python main.py --config configs/default.yaml
```

Each run trains all backbones in `model.backbones` sequentially. When finished, it:

1. Saves per-epoch training curves to `plots/{session}_{revision}_comparison.png`
2. Saves a test-accuracy summary chart to `plots/{session}_{revision}_summary.png`
3. Exports the best-performing backbone to `model/DeepScanClassifier.mlpackage`

To skip the Core ML export:

```bash
uv run python main.py --no-export
```

## Checkpoints

Each backbone run creates a timestamped directory:

```
checkpoints/{session}/{YYYY-MM-DD_HHMMSS}_{backbone}/
    best.ckpt     — best validation accuracy checkpoint
    config.yaml   — copy of the config used
    metrics.json  — per-epoch train/val metrics and final test accuracy
```

## Dataset

The dataset is downloaded automatically from [HuggingFace](https://huggingface.co/datasets/fish-gang/deepscan-dataset) on first run and cached under `data/`. It contains 7'794 images across 14 classes: 12 tropical reef fish species plus `unknown_fish` and `no_fish` rejection classes. The 70/15/15 train/val/test split is applied at training time (stratified by class, seed 42).

## Detector

`detector/` contains YOLO-World weights. `yolov8m-worldv2.pt` was exported as `model/FishDetector.mlpackage`.

To re-export the detector:

```bash
uv run python -m scripts.export_detector --export
```

## Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

```bash
docker build -t deepscan-model .
docker run --gpus all deepscan-model
```

## Long Training Runs

On the GPU server, use `screen` so training survives SSH disconnects:

```bash
screen -S deepscan
uv run python main.py
# Detach: Ctrl+A, then D
# Reconnect: screen -r deepscan
```
