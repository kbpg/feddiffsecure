from __future__ import annotations

from fnmatch import fnmatch
from typing import Dict

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """A tiny LoRA wrapper for Conv2d.

    The base convolution is frozen. Only the low-rank branch is trained.
    This is intentionally minimal so the whole project can run on a laptop.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rank: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.base = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)

        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, stride=stride, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


LO_PREFIXES = ("lora_down", "lora_up")


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, pattern) or name.startswith(pattern) for pattern in patterns)


def enable_parameter_prefixes(module: nn.Module, prefixes: list[str]) -> list[str]:
    if not prefixes:
        return []

    enabled: list[str] = []
    for name, param in module.named_parameters():
        if _matches_any_pattern(name, prefixes):
            param.requires_grad = True
            enabled.append(name)
    return enabled


def trainable_parameter_names(module: nn.Module) -> set[str]:
    return {name for name, param in module.named_parameters() if param.requires_grad}


def extract_trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    allowed_names = trainable_parameter_names(module)
    for name, tensor in module.state_dict().items():
        if name in allowed_names:
            output[name] = tensor.detach().cpu().clone()
    return output


def load_trainable_state_dict(module: nn.Module, trainable_state: Dict[str, torch.Tensor]) -> None:
    if not trainable_state:
        return

    allowed_names = trainable_parameter_names(module)
    missing, unexpected = module.load_state_dict(trainable_state, strict=False)
    bad_missing = [name for name in missing if name in allowed_names]
    if bad_missing:
        raise RuntimeError(f"Missing trainable keys: {bad_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected trainable keys: {unexpected}")


def average_state_dicts(states: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    if not weights:
        raise ValueError("weights must not be empty")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length")

    total = float(sum(weights))
    if total == 0.0:
        raise ValueError("sum of weights must be non-zero")

    averaged: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        acc = states[0][key].float() * (weights[0] / total)
        for state, weight in zip(states[1:], weights[1:]):
            acc = acc + state[key].float() * (weight / total)
        averaged[key] = acc.to(states[0][key].dtype)
    return averaged


def extract_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        if any(part in name for part in LO_PREFIXES):
            output[name] = tensor.detach().cpu().clone()
    return output


def load_lora_state_dict(module: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    if not lora_state:
        return
    missing, unexpected = module.load_state_dict(lora_state, strict=False)
    unexpected = [x for x in unexpected if any(part in x for part in LO_PREFIXES)]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")
    bad_missing = [x for x in missing if any(part in x for part in LO_PREFIXES)]
    if bad_missing:
        raise RuntimeError(f"Missing LoRA keys: {bad_missing}")


def average_lora_state_dicts(states: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    return average_state_dicts(states, weights)
