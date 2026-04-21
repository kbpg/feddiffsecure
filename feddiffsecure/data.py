from __future__ import annotations

import gzip
import pickle
import shutil
import struct
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


SUPPORTED_DATASETS = {"fashion_mnist", "mnist", "stl10", "cifar10", "fake_data"}


FASHION_MNIST_URLS = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

STL10_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
STL10_IMAGE_BYTES = {
    "train": 5000 * 3 * 96 * 96,
    "test": 8000 * 3 * 96 * 96,
}
STL10_LABEL_BYTES = {
    "train": 5000,
    "test": 8000,
}
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


class IndexedSubset(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], client_id: int) -> None:
        super().__init__(dataset, indices)
        self.client_id = client_id


class IDXDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.targets = labels.tolist()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.targets[idx])


class SimpleFakeDataset(Dataset):
    def __init__(self, size: int, channels: int, image_size: int, num_classes: int = 10, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, channels, image_size, image_size, generator=g) * 2.0 - 1.0
        self.targets = torch.randint(0, num_classes, (size,), generator=g).tolist()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.targets[idx])


def _download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        destination.unlink()
    if destination.exists():
        return
    urllib.request.urlretrieve(url, destination)


def _extract_tar_gz(archive_path: Path, destination_dir: Path, force: bool = False) -> None:
    if force and destination_dir.exists():
        shutil.rmtree(destination_dir)
    if destination_dir.exists():
        return
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=destination_dir.parent)


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(num, rows, cols)
    tensor = torch.from_numpy(data).float().unsqueeze(1) / 255.0
    return tensor * 2.0 - 1.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        if len(data) != num:
            raise ValueError(f"Label count mismatch in {path}")
    return torch.from_numpy(data).long()

def _load_idx_dataset(root: str | Path, urls: dict, train: bool) -> IDXDataset:
    root = Path(root)
    raw_dir = root / "raw"
    split = "train" if train else "test"
    image_key = f"{split}_images"
    label_key = f"{split}_labels"
    image_path = raw_dir / Path(urls[image_key]).name
    label_path = raw_dir / Path(urls[label_key]).name
    _download(urls[image_key], image_path)
    _download(urls[label_key], label_path)
    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)
    return IDXDataset(images=images, labels=labels)


def _read_stl10_images(path: Path) -> torch.Tensor:
    data = np.fromfile(path, dtype=np.uint8)
    images = data.reshape(-1, 3, 96, 96)
    images = np.transpose(images, (0, 1, 3, 2))
    tensor = torch.from_numpy(images.copy()).float() / 255.0
    return tensor * 2.0 - 1.0


def _read_stl10_labels(path: Path) -> torch.Tensor:
    labels = np.fromfile(path, dtype=np.uint8).astype(np.int64) - 1
    return torch.from_numpy(labels)


def _stl10_integrity_errors(extracted_dir: Path) -> list[str]:
    errors: list[str] = []
    for split, expected_bytes in STL10_IMAGE_BYTES.items():
        path = extracted_dir / f"{split}_X.bin"
        if not path.exists():
            errors.append(f"missing image file: {path}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"image file size mismatch for {path.name}: got {actual_bytes} bytes, expected {expected_bytes}"
            )

    for split, expected_bytes in STL10_LABEL_BYTES.items():
        path = extracted_dir / f"{split}_y.bin"
        if not path.exists():
            errors.append(f"missing label file: {path}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"label file size mismatch for {path.name}: got {actual_bytes} bytes, expected {expected_bytes}"
            )
    return errors


def _ensure_stl10_dataset(root: Path) -> Path:
    archive_path = root / Path(STL10_URL).name
    extracted_dir = root / "stl10_binary"

    _download(STL10_URL, archive_path)
    _extract_tar_gz(archive_path, extracted_dir)

    errors = _stl10_integrity_errors(extracted_dir)
    if not errors:
        return extracted_dir

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    if archive_path.exists():
        archive_path.unlink()

    _download(STL10_URL, archive_path, force=True)
    _extract_tar_gz(archive_path, extracted_dir, force=True)
    errors = _stl10_integrity_errors(extracted_dir)
    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise ValueError(
            "STL10 dataset files are corrupted or incomplete even after re-download.\n"
            "Please remove the dataset directory and download again.\n"
            f"{details}"
        )
    return extracted_dir


def _load_stl10_dataset(root: str | Path, train: bool, image_size: int, channels: int) -> IDXDataset:
    root = Path(root) / "stl10"
    extracted_dir = _ensure_stl10_dataset(root)

    split = "train" if train else "test"
    images = _read_stl10_images(extracted_dir / f"{split}_X.bin")
    labels = _read_stl10_labels(extracted_dir / f"{split}_y.bin")

    if int(image_size) != 96:
        images = F.interpolate(images, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)

    if int(channels) == 1:
        images = images.mean(dim=1, keepdim=True)
    elif int(channels) != 3:
        raise ValueError(f"Unsupported STL10 channel count: {channels}")

    return IDXDataset(images=images, labels=labels)


def _load_cifar10_dataset(root: str | Path, train: bool, image_size: int, channels: int) -> IDXDataset:
    root = Path(root) / "cifar10"
    archive_path = root / Path(CIFAR10_URL).name
    extracted_dir = root / "cifar-10-batches-py"
    _download(CIFAR10_URL, archive_path)
    _extract_tar_gz(archive_path, extracted_dir)

    batch_files = (
        [extracted_dir / f"data_batch_{idx}" for idx in range(1, 6)]
        if train
        else [extracted_dir / "test_batch"]
    )
    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for batch_path in batch_files:
        with open(batch_path, "rb") as file:
            batch = pickle.load(file, encoding="latin1")
        images_list.append(batch["data"])
        labels_list.extend(int(x) for x in batch["labels"])

    data = np.concatenate(images_list, axis=0).reshape(-1, 3, 32, 32)
    images = torch.from_numpy(data.copy()).float() / 255.0
    images = images * 2.0 - 1.0
    labels = torch.tensor(labels_list, dtype=torch.long)

    if int(image_size) != 32:
        images = F.interpolate(images, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)

    if int(channels) == 1:
        images = images.mean(dim=1, keepdim=True)
    elif int(channels) != 3:
        raise ValueError(f"Unsupported CIFAR10 channel count: {channels}")

    return IDXDataset(images=images, labels=labels)


def get_dataset(name: str, root: str, train: bool = True, image_size: int = 28, channels: int = 1) -> Dataset:
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")

    if name == "fake_data":
        size = 256 if train else 64
        return SimpleFakeDataset(size=size, channels=channels, image_size=image_size, seed=123 if train else 456)
    if name == "fashion_mnist":
        return _load_idx_dataset(root=Path(root) / "fashion_mnist", urls=FASHION_MNIST_URLS, train=train)
    if name == "mnist":
        return _load_idx_dataset(root=Path(root) / "mnist", urls=MNIST_URLS, train=train)
    if name == "stl10":
        return _load_stl10_dataset(root=root, train=train, image_size=image_size, channels=channels)
    if name == "cifar10":
        return _load_cifar10_dataset(root=root, train=train, image_size=image_size, channels=channels)
    raise ValueError(name)


def _extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
    elif hasattr(dataset, "labels"):
        targets = getattr(dataset, "labels")
    elif hasattr(dataset, "base_dataset"):
        return _extract_targets(getattr(dataset, "base_dataset"))
    elif isinstance(dataset, Subset):
        base_targets = _extract_targets(dataset.dataset)
        return np.array(base_targets)[dataset.indices]
    else:
        raise AttributeError("Dataset has neither targets nor labels attribute.")
    return np.array(targets)


def dirichlet_split(dataset: Dataset, num_clients: int, alpha: float, seed: int) -> List[IndexedSubset]:
    rng = np.random.default_rng(seed)
    targets = _extract_targets(dataset)
    num_classes = int(targets.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cut_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cut_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    subsets = []
    for client_id, indices in enumerate(client_indices):
        rng.shuffle(indices)
        subsets.append(IndexedSubset(dataset, indices, client_id=client_id))
    return subsets


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def summarize_client_splits(subsets: list[IndexedSubset]) -> list[dict]:
    summaries = []
    for subset in subsets:
        targets = _extract_targets(subset.dataset)[subset.indices]
        counts = Counter(targets.tolist())
        summaries.append({
            "client_id": subset.client_id,
            "num_samples": len(subset.indices),
            "class_histogram": dict(sorted(counts.items())),
        })
    return summaries
