from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from .lora import average_state_dicts, load_trainable_state_dict


@dataclass
class AggregationSummary:
    round_id: int
    num_clients: int
    avg_client_loss: float
    total_samples: int


def _save_tensor_grid(images: torch.Tensor, path: str, nrow: int = 2) -> None:
    images = images.detach().cpu().clamp(0.0, 1.0)
    b, c, h, w = images.shape
    nrow = max(1, min(nrow, b))
    ncol = int(np.ceil(b / nrow))
    canvas = torch.ones(c, ncol * h, nrow * w)
    for idx in range(b):
        r = idx // nrow
        col = idx % nrow
        canvas[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = images[idx]
    array = (canvas.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    scale = max(1, 112 // max(h, w))
    if array.shape[2] == 1:
        image = Image.fromarray(array[:, :, 0], mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    image.save(path)


class FederatedServer:
    def __init__(self, model: torch.nn.Module, diffusion, cfg: dict, device: str) -> None:
        self.model = model.to(device)
        self.diffusion = diffusion
        self.cfg = cfg
        self.device = device

    def get_global_model(self) -> torch.nn.Module:
        return copy.deepcopy(self.model)

    def aggregate(self, round_id: int, client_payloads: List[Dict]) -> AggregationSummary:
        states = [payload["trainable_state"] for payload in client_payloads]
        weights = [payload["num_samples"] for payload in client_payloads]
        avg_state = average_state_dicts(states, weights)
        load_trainable_state_dict(self.model, avg_state)
        avg_loss = sum(payload["avg_loss"] for payload in client_payloads) / max(len(client_payloads), 1)
        total_samples = int(sum(weights))
        return AggregationSummary(
            round_id=round_id,
            num_clients=len(client_payloads),
            avg_client_loss=avg_loss,
            total_samples=total_samples,
        )

    @torch.no_grad()
    def generate_samples(
        self,
        num_images: int = 4,
        sample_steps: int | None = None,
        sampler: str | None = None,
        ddim_eta: float | None = None,
        clip_denoised: bool | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = int(self.cfg["dataset"]["channels"])
        size = int(self.cfg["dataset"]["image_size"])
        diffusion_cfg = self.cfg.get("diffusion", {})
        steps = int(sample_steps if sample_steps is not None else diffusion_cfg.get("sample_steps", self.diffusion.timesteps))
        sample_sampler = str(sampler or diffusion_cfg.get("sampler", "ddpm"))
        sample_eta = float(diffusion_cfg.get("ddim_eta", 0.0) if ddim_eta is None else ddim_eta)
        sample_clip = bool(diffusion_cfg.get("clip_denoised", True) if clip_denoised is None else clip_denoised)
        samples = self.diffusion.sample(
            self.model,
            shape=(num_images, channels, size, size),
            device=self.device,
            sample_steps=steps,
            sampler=sample_sampler,
            ddim_eta=sample_eta,
            clip_denoised=sample_clip,
            initial_noise=initial_noise,
        )
        return samples

    @torch.no_grad()
    def save_samples(
        self,
        path: str,
        num_images: int = 4,
        sample_steps: int | None = None,
        sampler: str | None = None,
        ddim_eta: float | None = None,
        clip_denoised: bool | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> None:
        samples = self.generate_samples(
            num_images=num_images,
            sample_steps=sample_steps,
            sampler=sampler,
            ddim_eta=ddim_eta,
            clip_denoised=clip_denoised,
            initial_noise=initial_noise,
        )
        samples = (samples + 1.0) / 2.0
        _save_tensor_grid(samples, path, nrow=2)
