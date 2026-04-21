from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def collect_real_images(
    dataset: Dataset,
    max_samples: int,
    batch_size: int = 128,
    num_workers: int = 0,
) -> torch.Tensor:
    max_samples = max(int(max_samples), 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    batches: List[torch.Tensor] = []
    total = 0
    for images, _ in loader:
        take = min(images.size(0), max_samples - total)
        batches.append(images[:take].cpu())
        total += take
        if total >= max_samples:
            break

    if not batches:
        raise ValueError("No images were collected for reference statistics.")

    return torch.cat(batches, dim=0)


def _extract_proxy_features(images: torch.Tensor, feature_size: int = 8) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected BCHW images, got shape={tuple(images.shape)}")
    images = images.float()
    pooled = F.adaptive_avg_pool2d(images, output_size=(feature_size, feature_size))
    return pooled.flatten(start_dim=1)


def _covariance(features: torch.Tensor) -> torch.Tensor:
    features = features.double()
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    denom = max(features.size(0) - 1, 1)
    return centered.T @ centered / denom


def _matrix_sqrt_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    sym = (matrix + matrix.T) * 0.5
    eye = torch.eye(sym.size(0), dtype=sym.dtype, device=sym.device)
    jitter = 0.0

    for _ in range(7):
        try:
            adjusted = sym if jitter == 0.0 else sym + eye * jitter
            eigvals, eigvecs = torch.linalg.eigh(adjusted)
            eigvals = torch.clamp(eigvals, min=eps)
            sqrt_diag = torch.diag(torch.sqrt(eigvals))
            return eigvecs @ sqrt_diag @ eigvecs.T
        except RuntimeError:
            jitter = eps if jitter == 0.0 else jitter * 10.0

    # Fallback for severely ill-conditioned covariance products.
    adjusted = sym + eye * max(jitter, eps)
    u, singular_values, vh = torch.linalg.svd(adjusted, full_matrices=False)
    singular_values = torch.clamp(singular_values, min=eps)
    return (u * torch.sqrt(singular_values).unsqueeze(0)) @ vh


def frechet_distance(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    real_features = real_features.double()
    fake_features = fake_features.double()

    mu_r = real_features.mean(dim=0)
    mu_f = fake_features.mean(dim=0)
    cov_r = _covariance(real_features)
    cov_f = _covariance(fake_features)

    diff = mu_r - mu_f
    cov_r_sqrt = _matrix_sqrt_psd(cov_r)
    middle = cov_r_sqrt @ cov_f @ cov_r_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle)

    trace_term = torch.trace(cov_r + cov_f - 2.0 * middle_sqrt)
    score = diff.dot(diff) + trace_term
    return float(torch.clamp(score, min=0.0).item())


def compute_proxy_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_size: int = 8,
) -> float:
    real_features = _extract_proxy_features(real_images, feature_size=feature_size)
    fake_features = _extract_proxy_features(fake_images, feature_size=feature_size)
    return frechet_distance(real_features, fake_features)
