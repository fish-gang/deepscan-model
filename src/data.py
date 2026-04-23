from pathlib import Path

from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.transforms import train_transforms, val_transforms

load_dotenv()


class DeepScanDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"].convert("RGB")  # Ensure 3 channels
        label = item["label"]  # int class index

        if self.transform:
            image = self.transform(image)

        return image, label


class DeepScanDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dataset_cfg = config.dataset
        self.data_cfg = config.data

    def setup(self, stage=None):
        dataset = load_deepscan_dataset(
            repo_id=self.dataset_cfg.repo_id,
            revision=self.dataset_cfg.revision,
            cache_dir=Path(self.dataset_cfg.cache_dir),
            force_download=self.dataset_cfg.force_download,
        )

        # First split: train+val vs test
        split1 = dataset.train_test_split(
            test_size=self.data_cfg.test_split,
            seed=42,
            stratify_by_column="label",
        )
        # Second split: train vs val
        split2 = split1["train"].train_test_split(
            test_size=self.data_cfg.val_split,
            seed=42,
            stratify_by_column="label",
        )

        self.train_ds = DeepScanDataset(
            split2["train"], transform=train_transforms(self.data_cfg.image_size)
        )
        self.val_ds = DeepScanDataset(
            split2["test"], transform=val_transforms(self.data_cfg.image_size)
        )
        self.test_ds = DeepScanDataset(
            split1["test"], transform=val_transforms(self.data_cfg.image_size)
        )

        self.label_names = dataset.features["label"].names
        print(
            f"Split sizes: train={len(self.train_ds)}, "
            f"val={len(self.val_ds)}, test={len(self.test_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
        )


def load_deepscan_dataset(
    repo_id: str, revision: str, cache_dir: Path, force_download: bool = False
):
    """Load the DeepScan dataset, downloading and caching locally under data/."""
    version_file = cache_dir.parent / ".dataset_version"

    cached_version = version_file.read_text().strip() if version_file.exists() else None

    if not force_download and cached_version == revision and cache_dir.exists():
        print(f"Loading cached dataset ({revision}) from {cache_dir}")
        return load_from_disk(str(cache_dir))

    if force_download:
        print("Force download requested, re-downloading dataset...")
    elif cached_version and cached_version != revision:
        print(f"Version mismatch: cached={cached_version}, requested={revision}")
        print("Re-downloading dataset...")

    print(f"Downloading dataset {repo_id} ({revision})...")
    dataset = load_dataset(repo_id, split="train", revision=revision)

    # Cache to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    version_file.write_text(revision)
    print(f"Cached dataset ({revision}) to {cache_dir}")

    return dataset
