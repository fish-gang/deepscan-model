# DeepScan Classification Model

Image classification model for tropical reef fish species, built with PyTorch.

## Prerequisites 

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/fish-gang/deepscan-model.git
cd deepscan-model
uv sync
```

`uv sync` automatically creates a new virtual environment with the correct Python version.

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