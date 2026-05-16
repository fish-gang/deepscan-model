import json
import shutil
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam

from src.data import DeepScanDataModule
from src.model import create_model


class DeepScanClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, backbone: str, lr: float, pretrained: bool):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            num_classes=num_classes, backbone=backbone, pretrained=pretrained
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class MetricsLogger(pl.Callback):
    """Saves per-epoch train/val metrics to metrics.json in the run directory."""

    def __init__(self, save_path: Path, backbone: str):
        self.save_path = save_path
        self.backbone = backbone
        self.train_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_loss: list[float] = []
        self.val_acc: list[float] = []
        self.test_loss: float | None = None
        self.test_acc: float | None = None

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m:
            self.train_loss.append(float(m["train_loss"]))
            self.train_acc.append(float(m["train_acc"]))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self.val_loss.append(float(m["val_loss"]))
        self.val_acc.append(float(m["val_acc"]))
        self._save()

    def on_test_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.test_loss = float(m.get("test_loss", float("nan")))
        self.test_acc = float(m.get("test_acc", float("nan")))
        self._save()

    def _save(self):
        data: dict = {
            "backbone": self.backbone,
            "epochs": list(range(1, len(self.val_loss) + 1)),
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }
        if self.test_loss is not None:
            data["test_loss"] = self.test_loss
            data["test_acc"] = self.test_acc
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)


def train(config, config_path: str):
    """Run training with the given config dict.

    Creates a timestamped checkpoint directory and saves a copy
    of the config alongside the model checkpoints.
    """
    dataset_cfg = config.dataset
    model_cfg = config.model
    train_cfg = config.training

    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path("checkpoints") / f"{timestamp}_{model_cfg.backbone}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoints
    shutil.copy2(config_path, run_dir / "config.yaml")

    data = DeepScanDataModule(config)

    model = DeepScanClassifier(
        num_classes=dataset_cfg.num_classes,
        backbone=model_cfg.backbone,
        lr=train_cfg.lr,
        pretrained=model_cfg.pretrained,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    metrics_cb = MetricsLogger(
        save_path=run_dir / "metrics.json",
        backbone=model_cfg.backbone,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="",
        version="",
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=train_cfg.log_every_n_steps,
        callbacks=[checkpoint_cb, metrics_cb],
        logger=tb_logger,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    return run_dir
