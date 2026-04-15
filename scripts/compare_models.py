from pathlib import Path
import yaml
import torch

for run_dir in sorted(Path("checkpoints").iterdir()):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        continue

    with open(config_path) as f:
        config = yaml.safe_load(f)

    backbone = config["model"]["backbone"]

    # Find best checkpoint
    best = list(run_dir.glob("best-*.ckpt"))
    if not best:
        continue

    ckpt = torch.load(best[0], map_location="cpu", weights_only=True)
    param_count = sum(p.numel() for p in ckpt["state_dict"].values())
    file_size_mb = best[0].stat().st_size / 1024 / 1024

    # TODO: Don't store stats in filename
    # Extract val_acc from filename
    val_acc = best[0].stem.split("val_acc=")[-1] if "val_acc=" in best[0].stem else "?"

    print(f"{backbone:25s}  params={param_count/1e6:.1f}M  size={file_size_mb:.0f}MB  val_acc={val_acc}")