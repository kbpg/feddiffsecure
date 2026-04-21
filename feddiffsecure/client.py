from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data import build_loader
from .lora import extract_trainable_state_dict


@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    avg_loss: float
    trainable_state: Dict[str, torch.Tensor]


class FederatedClient:
    def __init__(self, client_id: int, subset, cfg: dict, diffusion, device: str) -> None:
        self.client_id = client_id
        self.subset = subset
        self.cfg = cfg
        self.diffusion = diffusion
        self.device = device

    def train(self, global_model: torch.nn.Module) -> ClientResult:
        model = copy.deepcopy(global_model).to(self.device)
        train_loader = build_loader(
            self.subset,
            batch_size=int(self.cfg["training"]["batch_size"]),
            num_workers=int(self.cfg["training"]["num_workers"]),
            shuffle=True,
        )
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(self.cfg["training"]["lr"]),
            weight_decay=float(self.cfg["training"]["weight_decay"]),
        )
        local_epochs = int(self.cfg["federated"]["local_epochs"])
        max_steps_per_epoch = int(self.cfg["federated"]["max_steps_per_epoch"])
        x0_loss_weight = float(self.cfg.get("diffusion", {}).get("x0_loss_weight", 0.0))

        model.train()
        total_loss = 0.0
        total_steps = 0
        progress = tqdm(range(local_epochs), desc=f"client {self.client_id}", leave=False)
        for _ in progress:
            for step, (x, _) in enumerate(train_loader):
                if step >= max_steps_per_epoch:
                    break
                x = x.to(self.device)
                t = torch.randint(0, self.diffusion.timesteps, (x.size(0),), device=self.device).long()
                loss = self.diffusion.p_losses(model, x, t, x0_loss_weight=x0_loss_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_steps += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

        return ClientResult(
            client_id=self.client_id,
            num_samples=len(self.subset),
            avg_loss=total_loss / max(total_steps, 1),
            trainable_state=extract_trainable_state_dict(model),
        )
