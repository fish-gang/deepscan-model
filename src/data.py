from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.transforms import train_transforms, val_transforms


DATASET_ID = "fish-gang/deepscan-dataset"
DATASET_REVISION = "v0.1"
DATA_DIR = Path("data")
VERSION_FILE = DATA_DIR / ".dataset_version"


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
    def __init__(
        self,
        revision=DATASET_REVISION,
        batch_size=32,
        num_workers=4,
        val_split=0.15,
        test_split=0.15,
        image_size=224,
        force_download=False,
    ):
        super().__init__()
        self.revision = revision
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.force_download = force_download

    def setup(self, stage=None):
        dataset = load_deepscan_dataset(
            revision=self.revision, force_download=self.force_download
        )

        # First split: train+val vs test
        split1 = dataset.train_test_split(
            test_size=self.test_split,
            seed=42,
            stratify_by_column="label",
        )
        # Second split: train vs val
        split2 = split1["train"].train_test_split(
            test_size=self.val_split,
            seed=42,
            stratify_by_column="label",
        )

        self.train_ds = DeepScanDataset(
            split2["train"], transform=train_transforms(self.image_size)
        )
        self.val_ds = DeepScanDataset(
            split2["test"], transform=val_transforms(self.image_size)
        )
        self.test_ds = DeepScanDataset(
            split1["test"], transform=val_transforms(self.image_size)
        )

        self.label_names = dataset.features["label"].names
        print(
            f"Split sizes: train={len(self.train_ds)}, "
            f"val={len(self.val_ds)}, test={len(self.test_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )


def _get_cached_version():
    """Read the cached dataset version from the version file."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    return None


def _write_cached_version(revision):
    """Write the dataset version to the version file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(revision)


def load_deepscan_dataset(revision=DATASET_REVISION, force_download=False):
    """Load the DeepScan dataset, downloading and caching locally under data/."""
    cached_version = _get_cached_version()
    cache_path = DATA_DIR / "hf_cache"

    if not force_download and cached_version == revision and cache_path.exists():
        print(f"Loading cached dataset ({revision}) from {cache_path}")
        from datasets import load_from_disk

        return load_from_disk(str(cache_path))

    if cached_version and cached_version != revision:
        print(f"Version mismatch: cached={cached_version}, requested={revision}")
        print("Re-downloading dataset...")

    print(f"Downloading dataset {DATASET_ID} ({revision})...")
    dataset = load_dataset(DATASET_ID, split="train", revision=revision)

    # Cache to disk
    dataset.save_to_disk(str(cache_path))
    _write_cached_version(revision)
    print(f"Cached dataset ({revision}) to {cache_path}")

    return dataset
