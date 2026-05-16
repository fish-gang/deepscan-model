from pathlib import Path
import json
import torch
import yaml

for run_dir in sorted(Path("checkpoints").iterdir()):
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        continue

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    backbone = config["model"]["backbone"]
    val_acc = max(metrics["val_acc"])
    test_acc = metrics.get("test_acc", float("nan"))

    best = list(run_dir.glob("best*.ckpt"))
    if not best:
        continue

    param_count = sum(
        p.numel()
        for p in torch.load(best[0], map_location="cpu", weights_only=True)["state_dict"].values()
    )
    file_size_mb = best[0].stat().st_size / 1024 / 1024

    print(
        f"{backbone:25s}  params={param_count / 1e6:.1f}M  "
        f"size={file_size_mb:.0f}MB  val_acc={val_acc:.1%}  test_acc={test_acc:.1%}"
    )