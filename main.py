import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DeepScanDataModule
from src.model import create_model


class DeepScanClassifier(pl.LightningModule):
    def __init__(self, num_classes=12, backbone="efficientnet_b0", lr=1e-3, pretrained=True):
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


def main():
    data = DeepScanDataModule(batch_size=32, num_workers=4)

    model = DeepScanClassifier(
        num_classes=12,
        backbone="efficientnet_b0",
        lr=1e-3,
        pretrained=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="deepscan-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # picks GPU if available, else CPU
        devices=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
