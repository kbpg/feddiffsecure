from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import feddiffsecure.runtime_env  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPTextModel, CLIPTokenizer

from feddiffsecure.audit import AuditLogger, summarize_audit_log
from feddiffsecure.data import dirichlet_split, get_dataset
from feddiffsecure.stego import package_state_as_stego_png, recover_state_from_stego_png
from feddiffsecure.utils import ensure_dir, load_yaml, resolve_device, save_json, set_seed


STL10_CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MNIST_CLASS_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

FASHION_MNIST_CLASS_NAMES = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]


@dataclass
class ClientResult:
    client_id: int
    avg_loss: float
    num_samples: int
    lora_state: dict[str, torch.Tensor]
    skipped_steps: int


class CaptionedTensorDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        dataset_name: str,
        resolution: int,
        caption_mode: str = "auto",
        caption_dropout: float = 0.0,
    ) -> None:
        self.base_dataset = base_dataset
        self.dataset_name = str(dataset_name).lower()
        self.resolution = int(resolution)
        self.caption_mode = str(caption_mode)
        self.caption_dropout = float(caption_dropout)
        if hasattr(base_dataset, "targets"):
            self.targets = list(getattr(base_dataset, "targets"))
        elif hasattr(base_dataset, "labels"):
            self.targets = list(getattr(base_dataset, "labels"))
        else:
            # Keep the wrapped dataset compatible with existing split helpers.
            self.targets = [int(base_dataset[idx][1]) for idx in range(len(base_dataset))]
        self.labels = self.targets

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _class_name_from_label(self, label: int) -> str:
        label = int(label)
        if self.dataset_name == "stl10":
            names = STL10_CLASS_NAMES
        elif self.dataset_name == "cifar10":
            names = CIFAR10_CLASS_NAMES
        elif self.dataset_name == "mnist":
            names = MNIST_CLASS_NAMES
        elif self.dataset_name == "fashion_mnist":
            names = FASHION_MNIST_CLASS_NAMES
        else:
            names = []
        return names[label] if 0 <= label < len(names) else "object"

    def _caption_from_label(self, label: int) -> str:
        label_name = self._class_name_from_label(label)
        if self.caption_mode in {"auto", "class_prompt"}:
            if self.dataset_name == "mnist":
                return f"a handwritten digit {label_name}"
            if self.dataset_name == "fashion_mnist":
                return f"a product photo of a {label_name}"
            return f"a photo of a {label_name}"
        if self.caption_mode == "mnist_class_prompt":
            return f"a handwritten digit {label_name}"
        if self.caption_mode == "fashion_mnist_class_prompt":
            return f"a product photo of a {label_name}"
        if self.caption_mode == "stl10_class_prompt":
            return f"a photo of a {label_name}"
        return "a natural image"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.base_dataset[idx]
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] != self.resolution or image.shape[-2] != self.resolution:
            image = F.interpolate(
                image,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
        image = image.squeeze(0).clamp(-1.0, 1.0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        caption = self._caption_from_label(int(label))
        if self.caption_dropout > 0.0 and random.random() < self.caption_dropout:
            caption = ""
        return {
            "pixel_values": image,
            "caption": caption,
            "label": int(label),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated SD1.5 LoRA training with frozen base weights.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def _write_round_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _average_state_dicts(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("sum of weights must be > 0")
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        acc = states[0][key].float() * (weights[0] / total_weight)
        for state, weight in zip(states[1:], weights[1:]):
            acc = acc + state[key].float() * (weight / total_weight)
        averaged[key] = acc.to(dtype=states[0][key].dtype)
    return averaged


def _build_client_subsets(dataset: Dataset, cfg: dict) -> list[Subset]:
    num_clients = int(cfg["federated"]["num_clients"])
    alpha = float(cfg["federated"].get("dirichlet_alpha", 0.0))
    if alpha > 0.0:
        return dirichlet_split(dataset, num_clients=num_clients, alpha=alpha, seed=int(cfg["seed"]))

    indices = np.array_split(np.arange(len(dataset)), num_clients)
    subsets: list[Subset] = []
    for client_id, split in enumerate(indices):
        subset = Subset(dataset, split.tolist())
        subset.client_id = client_id  # type: ignore[attr-defined]
        subsets.append(subset)
    return subsets


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "captions": [item["caption"] for item in batch],
        "labels": [item["label"] for item in batch],
    }


def _make_scheduler(scheduler_name: str, pretrained_model_name_or_path: str, token: str | None = None):
    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == "ddpm":
        return DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", token=token)
    if scheduler_name == "ddim":
        return DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", token=token)
    if scheduler_name in {"dpmpp", "dpmpp_2m", "dpm_solver"}:
        base = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", token=token)
        return DPMSolverMultistepScheduler.from_config(base.config)
    raise ValueError(f"Unsupported sampling scheduler: {scheduler_name}")


def _resolve_torch_dtype(mixed_precision: str, device: str) -> torch.dtype:
    precision = str(mixed_precision).lower()
    if not device.startswith("cuda"):
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _cast_trainable_params_to_fp32(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def _state_l2_norm(state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in state.values():
        total += float(torch.sum(tensor.detach().float() ** 2).item())
    return total**0.5


def _state_delta_l2_norm(state_a: dict[str, torch.Tensor], state_b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in state_a.keys():
        delta = state_a[key].detach().float() - state_b[key].detach().float()
        total += float(torch.sum(delta ** 2).item())
    return total**0.5


def _state_numel(state: dict[str, torch.Tensor]) -> int:
    return int(sum(int(tensor.numel()) for tensor in state.values()))


def _make_pipeline(
    pretrained_model_name_or_path: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler_name: str,
    torch_dtype: torch.dtype,
    device: str,
    token: str | None = None,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=torch_dtype,
        token=token,
    )
    pipe.scheduler = _make_scheduler(scheduler_name, pretrained_model_name_or_path, token=token)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


def _save_pil_grid(images: list[Image.Image], path: Path, columns: int = 2) -> None:
    if not images:
        raise ValueError("images must not be empty")
    columns = max(1, min(int(columns), len(images)))
    width, height = images[0].size
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), color="white")
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        canvas.paste(image, (col * width, row * height))
    canvas.save(path)


def _sample_and_save(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    negative_prompt: str,
    cfg: dict,
    output_path: Path,
    seed: int,
) -> None:
    sampling_cfg = cfg["sampling"]
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    images = pipe(
        prompt=prompts,
        negative_prompt=[negative_prompt] * len(prompts) if negative_prompt else None,
        num_inference_steps=int(sampling_cfg["num_inference_steps"]),
        guidance_scale=float(sampling_cfg["guidance_scale"]),
        height=int(sampling_cfg["height"]),
        width=int(sampling_cfg["width"]),
        generator=generator,
    ).images
    _save_pil_grid(images, output_path, columns=min(2, len(images)))


def _train_one_client(
    subset: Subset,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    cfg: dict,
    device: str,
    torch_dtype: torch.dtype,
) -> ClientResult:
    training_cfg = cfg["training"]
    batch_size = int(training_cfg["batch_size"])
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    client_epochs = int(cfg["federated"].get("client_epochs", 1))
    max_steps_per_client = int(cfg["federated"].get("max_steps_per_client", 0))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        drop_last=True,
        collate_fn=_collate_batch,
    )
    optimizer = AdamW(
        [parameter for parameter in unet.parameters() if parameter.requires_grad],
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        eps=float(training_cfg.get("optimizer_eps", 1e-8)),
    )

    unet.train()
    total_loss = 0.0
    total_steps = 0
    skipped_steps = 0
    optimizer.zero_grad(set_to_none=True)
    def _autocast_ctx():
        if device.startswith("cuda") and torch_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=torch_dtype)
        return nullcontext()
    vae_dtype = next(vae.parameters()).dtype
    for _ in range(client_epochs):
        for step, batch in enumerate(loader):
            if max_steps_per_client > 0 and total_steps >= max_steps_per_client:
                break
            pixel_values = batch["pixel_values"].to(device=device, dtype=vae_dtype)
            input_ids = tokenizer(
                batch["captions"],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                with _autocast_ctx():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

            if not torch.isfinite(latents).all():
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            with _autocast_ctx():
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            if not torch.isfinite(model_pred).all():
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue

            prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_type: {prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue
            (loss / grad_accum_steps).backward()
            total_loss += float(loss.item())
            total_steps += 1

            if total_steps % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in unet.parameters() if parameter.requires_grad],
                    max_norm=float(training_cfg.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if max_steps_per_client > 0 and total_steps >= max_steps_per_client:
            break

    if total_steps % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in unet.parameters() if parameter.requires_grad],
            max_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    client_id = int(getattr(subset, "client_id", -1))
    return ClientResult(
        client_id=client_id,
        avg_loss=(total_loss / max(total_steps, 1)) if total_steps > 0 else float("inf"),
        num_samples=len(subset),
        lora_state={name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(unet).items()},
        skipped_steps=skipped_steps,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    run_started_at = datetime.now().astimezone().isoformat()
    device = resolve_device(str(cfg["system"].get("device", "auto")))
    torch_dtype = _resolve_torch_dtype(str(cfg["training"].get("mixed_precision", "fp16")), device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = ensure_dir(cfg["system"]["output_dir"])
    samples_dir = ensure_dir(output_dir / "samples")
    metrics_dir = ensure_dir(output_dir / "metrics")
    carriers_dir = ensure_dir(output_dir / "audit_carriers")
    token = cfg.get("model", {}).get("hf_token") or os.environ.get("HF_TOKEN") or None
    audit_cfg = cfg.get("audit", {})
    audit_enabled = bool(audit_cfg.get("enabled", True))
    audit_logger = AuditLogger(output_dir / "audit_log.jsonl")
    save_carriers = bool(audit_cfg.get("save_carriers", False))
    audit_key_prefix = str(audit_cfg.get("key_prefix", "tiny-sd-lora"))

    pretrained_model_name_or_path = str(cfg["model"]["pretrained_model_name_or_path"])
    dataset_name = str(cfg["dataset"]["name"]).lower()
    resolution = int(cfg["dataset"].get("resolution", 512))
    base_dataset = get_dataset(
        name=dataset_name,
        root=str(cfg["dataset"]["root"]),
        train=True,
        image_size=96 if dataset_name == "stl10" else resolution,
        channels=3,
    )
    train_dataset = CaptionedTensorDataset(
        base_dataset=base_dataset,
        dataset_name=dataset_name,
        resolution=resolution,
        caption_mode=str(cfg["dataset"].get("caption_mode", "auto")),
        caption_dropout=float(cfg["dataset"].get("caption_dropout", 0.0)),
    )
    client_subsets = _build_client_subsets(train_dataset, cfg)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", token=token)
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        token=token,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
        token=token,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch_dtype,
        token=token,
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", token=token)

    if bool(cfg["training"].get("gradient_checkpointing", False)):
        unet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["rank"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        lora_dropout=float(cfg["lora"].get("dropout", 0.0)),
        bias="none",
    )
    unet.add_adapter(lora_cfg)
    _cast_trainable_params_to_fp32(unet)
    global_lora_state = {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(unet).items()}
    lora_numel = _state_numel(global_lora_state)

    rounds = int(cfg["federated"]["rounds"])
    clients_per_round = int(cfg["federated"]["clients_per_round"])
    eval_every = int(cfg["sampling"].get("eval_every", 5))
    sample_seed = int(cfg["sampling"].get("seed", cfg["seed"]))
    prompts = list(cfg["sampling"]["prompts"])
    negative_prompt = str(cfg["sampling"].get("negative_prompt", ""))

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "started_at": run_started_at,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "dataset": cfg["dataset"]["name"],
        "resolution": resolution,
        "training_mode": "federated_lora_sd15",
        "device": device,
        "dtype": str(torch_dtype),
        "rounds": rounds,
        "num_clients": int(cfg["federated"]["num_clients"]),
        "clients_per_round": clients_per_round,
        "client_epochs": int(cfg["federated"].get("client_epochs", 1)),
        "max_steps_per_client": int(cfg["federated"].get("max_steps_per_client", 0)),
        "lora_rank": int(cfg["lora"]["rank"]),
        "lora_alpha": int(cfg["lora"]["alpha"]),
        "target_modules": list(cfg["lora"]["target_modules"]),
        "sample_seed": sample_seed,
        "train_examples": len(train_dataset),
        "lora_numel": lora_numel,
        "audit_enabled": audit_enabled,
    }
    save_json(summary, output_dir / "run_summary.json")

    best_loss = float("inf")
    round_metrics: list[dict[str, Any]] = []

    for round_id in range(1, rounds + 1):
        round_start = time.perf_counter()
        selected_subsets = random.sample(client_subsets, k=min(clients_per_round, len(client_subsets)))
        client_results: list[ClientResult] = []
        client_payloads = []
        client_update_norms: list[float] = []
        payload_bytes_raw: list[int] = []
        payload_bytes_compressed: list[int] = []
        audit_verified_count = 0

        for subset in selected_subsets:
            set_peft_model_state_dict(unet, global_lora_state, adapter_name="default")
            result = _train_one_client(
                subset=subset,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                noise_scheduler=noise_scheduler,
                cfg=cfg,
                device=device,
                torch_dtype=torch_dtype,
            )
            client_results.append(result)
            client_update_norm = _state_delta_l2_norm(result.lora_state, global_lora_state)
            client_update_norms.append(client_update_norm)

            effective_state = result.lora_state
            if audit_enabled:
                key = f"{audit_key_prefix}-round{round_id}-client{result.client_id}"
                carrier_path = carriers_dir / f"round_{round_id:03d}_client_{result.client_id:02d}.png"
                carrier_path, package_info = package_state_as_stego_png(
                    result.lora_state,
                    output_path=carrier_path,
                    key=key,
                    seed=int(cfg["seed"]) + round_id * 100 + result.client_id,
                )
                recovered_state, recover_info = recover_state_from_stego_png(carrier_path, key=key)
                verified = bool(package_info["raw_sha256"] == recover_info["raw_sha256"])
                effective_state = recovered_state
                payload_bytes_raw.append(int(package_info["payload_bytes_raw"]))
                payload_bytes_compressed.append(int(package_info["payload_bytes_compressed"]))
                audit_verified_count += int(verified)
                carrier_record_path = str(carrier_path) if save_carriers else ""
                if not save_carriers and carrier_path.exists():
                    carrier_path.unlink()
                audit_logger.log(
                    {
                        "round": round_id,
                        "client_id": result.client_id,
                        "num_samples": result.num_samples,
                        "avg_loss": result.avg_loss,
                        "skipped_steps": result.skipped_steps,
                        "carrier_path": carrier_record_path,
                        "carrier_saved": save_carriers,
                        "carrier_psnr": package_info["psnr"],
                        "payload_bytes_raw": package_info["payload_bytes_raw"],
                        "payload_bytes_compressed": package_info["payload_bytes_compressed"],
                        "state_sha256": package_info["raw_sha256"],
                        "recovered_sha256": recover_info["raw_sha256"],
                        "verified": verified,
                    }
                )

            client_payloads.append(
                {
                    "client_id": result.client_id,
                    "num_samples": result.num_samples,
                    "avg_loss": result.avg_loss,
                    "trainable_state": effective_state,
                }
            )

        averaged_state = _average_state_dicts(
            [payload["trainable_state"] for payload in client_payloads],
            [payload["num_samples"] for payload in client_payloads],
        )
        aggregation_delta_l2 = _state_delta_l2_norm(averaged_state, global_lora_state)
        set_peft_model_state_dict(unet, averaged_state, adapter_name="default")
        global_lora_state = {name: tensor.detach().cpu().clone() for name, tensor in averaged_state.items()}

        avg_client_loss = sum(result.avg_loss for result in client_results) / max(len(client_results), 1)
        total_samples = int(sum(result.num_samples for result in client_results))
        total_skipped_steps = int(sum(result.skipped_steps for result in client_results))
        round_record = {
            "round": round_id,
            "avg_client_loss": round(avg_client_loss, 6),
            "selected_clients": [result.client_id for result in client_results],
            "total_samples": total_samples,
            "skipped_steps": total_skipped_steps,
            "mean_client_update_l2": round(float(sum(client_update_norms) / max(len(client_update_norms), 1)), 6),
            "max_client_update_l2": round(float(max(client_update_norms) if client_update_norms else 0.0), 6),
            "aggregation_delta_l2": round(float(aggregation_delta_l2), 6),
            "global_lora_l2": round(float(_state_l2_norm(global_lora_state)), 6),
            "payload_bytes_raw_sum": int(sum(payload_bytes_raw)),
            "payload_bytes_compressed_sum": int(sum(payload_bytes_compressed)),
            "payload_bytes_raw_mean": round(float(sum(payload_bytes_raw) / max(len(payload_bytes_raw), 1)), 3),
            "payload_bytes_compressed_mean": round(float(sum(payload_bytes_compressed) / max(len(payload_bytes_compressed), 1)), 3),
            "audit_verified": int(audit_verified_count),
            "audit_records": int(len(payload_bytes_raw)),
            "round_duration_sec": round(time.perf_counter() - round_start, 3),
        }
        round_metrics.append(round_record)
        save_json(round_metrics, metrics_dir / "round_metrics.json")
        _write_round_metrics_csv(round_metrics, metrics_dir / "round_metrics.csv")
        torch.save(global_lora_state, output_dir / "global_lora_latest.pt")

        if avg_client_loss < best_loss:
            best_loss = avg_client_loss
            torch.save(global_lora_state, output_dir / "global_lora_best_loss.pt")

        if round_id % eval_every == 0 or round_id == rounds:
            pipe = _make_pipeline(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                scheduler_name=str(cfg["sampling"].get("scheduler", "dpmpp_2m")),
                torch_dtype=torch_dtype,
                device=device,
                token=token,
            )
            _sample_and_save(
                pipe=pipe,
                prompts=prompts,
                negative_prompt=negative_prompt,
                cfg=cfg,
                output_path=samples_dir / f"round_{round_id:03d}.png",
                seed=sample_seed,
            )
            del pipe

        print(
            f"[Round {round_id}] avg_loss={avg_client_loss:.4f}, "
            f"selected_clients={[result.client_id for result in client_results]}, "
            f"total_samples={total_samples}, skipped_steps={total_skipped_steps}, "
            f"audit={audit_verified_count}/{len(payload_bytes_raw) if audit_enabled else 0}"
        )

    torch.save(global_lora_state, output_dir / "global_lora_final.pt")
    set_peft_model_state_dict(unet, global_lora_state, adapter_name="default")
    final_pipe = _make_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler_name=str(cfg["sampling"].get("scheduler", "dpmpp_2m")),
        torch_dtype=torch_dtype,
        device=device,
        token=token,
    )
    _sample_and_save(
        pipe=final_pipe,
        prompts=prompts,
        negative_prompt=negative_prompt,
        cfg=cfg,
        output_path=samples_dir / "final.png",
        seed=sample_seed,
    )
    del final_pipe

    if (output_dir / "global_lora_best_loss.pt").exists():
        set_peft_model_state_dict(
            unet,
            torch.load(output_dir / "global_lora_best_loss.pt", map_location="cpu"),
            adapter_name="default",
        )
        best_pipe = _make_pipeline(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler_name=str(cfg["sampling"].get("scheduler", "dpmpp_2m")),
            torch_dtype=torch_dtype,
            device=device,
            token=token,
        )
        _sample_and_save(
            pipe=best_pipe,
            prompts=prompts,
            negative_prompt=negative_prompt,
            cfg=cfg,
            output_path=samples_dir / "best_loss.png",
            seed=sample_seed,
        )
        del best_pipe

    final_metrics = {
        "last_round": round_metrics[-1] if round_metrics else {},
        "best_loss_round": min(round_metrics, key=lambda item: item["avg_client_loss"]) if round_metrics else {},
        "best_loss": round(float(best_loss), 6) if round_metrics else None,
        "global_lora_final_path": str((output_dir / "global_lora_final.pt").resolve()),
        "global_lora_best_loss_path": str((output_dir / "global_lora_best_loss.pt").resolve()),
    }
    save_json(final_metrics, metrics_dir / "final_metrics.json")
    audit_summary = summarize_audit_log(output_dir / "audit_log.jsonl")
    save_json(audit_summary, output_dir / "audit_summary.json")

    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
