import shutil
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
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
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
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
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=train_cfg.log_every_n_steps,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
