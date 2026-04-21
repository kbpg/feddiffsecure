from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SUPPORTED_DATASETS, get_dataset
from .metrics import frechet_distance
from .utils import ensure_dir, save_json, set_seed


EVALUATOR_VERSION = 2
SUPPORTED_SEMANTIC_DATASETS = {name for name in SUPPORTED_DATASETS if name != "fake_data"}


class LabelFeatureClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() + 1.0) * 0.5
        h = self.features(x)
        h = self.pool(h)
        return self.embedding(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)


@dataclass
class SemanticEvaluatorInfo:
    dataset_name: str
    checkpoint_path: str
    num_classes: int
    feature_dim: int
    test_accuracy: float


def supports_semantic_metrics(dataset_name: str) -> bool:
    return str(dataset_name).lower() in SUPPORTED_SEMANTIC_DATASETS


def _dataset_num_classes(dataset) -> int:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset does not expose targets for evaluator training.")
    if isinstance(targets, list):
        return int(max(targets)) + 1
    return int(torch.as_tensor(targets).max().item()) + 1


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def _evaluate_classifier(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def load_or_train_label_evaluator(
    dataset_name: str,
    dataset_root: str,
    image_size: int,
    channels: int,
    device: str,
    cache_dir: str | Path,
    num_workers: int = 0,
    batch_size: int = 256,
    epochs: int = 5,
    force_retrain: bool = False,
) -> tuple[LabelFeatureClassifier, SemanticEvaluatorInfo]:
    dataset_name = str(dataset_name).lower()
    if not supports_semantic_metrics(dataset_name):
        raise ValueError(f"Semantic evaluator is not supported for dataset={dataset_name}")

    cache_dir = ensure_dir(cache_dir)
    checkpoint_path = cache_dir / f"{dataset_name}_label_classifier_v{EVALUATOR_VERSION}.pt"
    metadata_path = cache_dir / f"{dataset_name}_label_classifier_v{EVALUATOR_VERSION}.json"

    if checkpoint_path.exists() and not force_retrain:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = LabelFeatureClassifier(
            in_channels=int(checkpoint["in_channels"]),
            num_classes=int(checkpoint["num_classes"]),
            feature_dim=int(checkpoint["feature_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        info = SemanticEvaluatorInfo(
            dataset_name=dataset_name,
            checkpoint_path=str(checkpoint_path.resolve()),
            num_classes=int(checkpoint["num_classes"]),
            feature_dim=int(checkpoint["feature_dim"]),
            test_accuracy=float(checkpoint["test_accuracy"]),
        )
        return model, info

    set_seed(1234)
    train_dataset = get_dataset(
        name=dataset_name,
        root=dataset_root,
        train=True,
        image_size=image_size,
        channels=channels,
    )
    test_dataset = get_dataset(
        name=dataset_name,
        root=dataset_root,
        train=False,
        image_size=image_size,
        channels=channels,
    )
    num_classes = _dataset_num_classes(train_dataset)
    model = LabelFeatureClassifier(in_channels=channels, num_classes=num_classes, feature_dim=128).to(device)

    train_loader = _make_loader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = _make_loader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    progress = tqdm(range(1, epochs + 1), desc=f"train {dataset_name} evaluator", leave=False)
    for epoch in progress:
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size_current = int(labels.numel())
            total_loss += float(loss.item()) * batch_size_current
            total_examples += batch_size_current

        test_acc = _evaluate_classifier(model, test_loader, device)
        avg_loss = total_loss / max(total_examples, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", test_acc=f"{test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Failed to train the semantic evaluator.")

    model.load_state_dict(best_state)
    checkpoint = {
        "dataset_name": dataset_name,
        "in_channels": channels,
        "num_classes": num_classes,
        "feature_dim": 128,
        "test_accuracy": float(best_acc),
        "model_state_dict": best_state,
    }
    torch.save(checkpoint, checkpoint_path)
    save_json(
        {
            "dataset_name": dataset_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "num_classes": num_classes,
            "feature_dim": 128,
            "evaluator_version": EVALUATOR_VERSION,
            "test_accuracy": float(best_acc),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
        },
        metadata_path,
    )

    info = SemanticEvaluatorInfo(
        dataset_name=dataset_name,
        checkpoint_path=str(checkpoint_path.resolve()),
        num_classes=num_classes,
        feature_dim=128,
        test_accuracy=float(best_acc),
    )
    return model, info


def _batched_features_and_probs(
    model: LabelFeatureClassifier,
    images: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size].to(device)
            batch_features = model.extract_features(batch)
            logits = model.classifier(batch_features)
            features.append(batch_features.cpu())
            probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(features, dim=0), torch.cat(probs, dim=0)


def compute_semantic_metrics(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    dataset_name: str,
    dataset_root: str,
    image_size: int,
    channels: int,
    device: str,
    cache_dir: str | Path,
    eval_batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, float | int] | None:
    dataset_name = str(dataset_name).lower()
    if not supports_semantic_metrics(dataset_name):
        return None

    evaluator, info = load_or_train_label_evaluator(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        image_size=int(image_size),
        channels=int(channels),
        device=device,
        cache_dir=cache_dir,
        num_workers=int(num_workers),
        batch_size=max(int(eval_batch_size), 64),
    )
    real_features, _ = _batched_features_and_probs(
        evaluator,
        real_images,
        device=device,
        batch_size=int(eval_batch_size),
    )
    fake_features, fake_probs = _batched_features_and_probs(
        evaluator,
        fake_images,
        device=device,
        batch_size=int(eval_batch_size),
    )

    classifier_fid = frechet_distance(real_features, fake_features)
    confidences, preds = fake_probs.max(dim=1)
    class_hist = torch.bincount(preds, minlength=info.num_classes).float()
    class_dist = class_hist / torch.clamp(class_hist.sum(), min=1.0)
    non_zero = class_dist > 0
    class_entropy = float(-(class_dist[non_zero] * class_dist[non_zero].log()).sum().item())

    return {
        "classifier_fid": float(classifier_fid),
        "top1_confidence_mean": float(confidences.mean().item()),
        "top1_confident_ratio": float((confidences >= 0.9).float().mean().item()),
        "predicted_class_entropy": class_entropy,
        "predicted_unique_classes": int((class_hist > 0).sum().item()),
        "evaluator_test_accuracy": float(info.test_accuracy),
    }
