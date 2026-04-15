# DeepScan Classification Model

Image classification model for tropical reef fish species, built with PyTorch.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/fish-gang/deepscan-model.git
cd deepscan-model
uv sync
```

`uv sync` automatically creates a new virtual environment with the correct Python version.

## Configuration

Training hyperparameters are defined in YAML config files under `configs/`. Run a custom experiment:

```bash
uv run python main.py --config configs/mobilenet_experiment.yaml
```

Each training run creates a timestamped directory under `checkpoints/` containing the config, best model, and last checkpoint.

## Running

```bash
uv run python main.py
```

For long training runs on the server, use tmux so the process survives disconnects:

```bash
tmux new -s training
uv run python main.py
# Detach: Ctrl+B, then D
# Reconnect: tmux attach -t training
```

## Docker

As an alternative to installing dependencies locally, you can use Docker. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

```bash
docker build -t deepscan-model .
docker run --gpus all deepscan-model --config configs/default.yaml
```

## Dataset

The dataset is loaded automatically from [HuggingFace](https://huggingface.co/datasets/fish-gang/deepscan-dataset) on first run and cached locally under `data/`. To force re-download, set `force_download=True`.

## Prediction

Run inference on new images using a trained checkpoint:

```bash
# Single image
uv run python predict.py --checkpoint checkpoints/<run>/best.ckpt --image fish.jpg

# Folder of images
uv run python predict.py --checkpoint checkpoints/<run>/best.ckpt --image images/
```

## Exporting to Core ML Format

Use the `scripts/export_coreml.py` script to export any model checkpoint to Core ML format:

```bash
uv run python -m scripts.export_coreml --checkpoint checkpoints/<run>/best.ckpt
```