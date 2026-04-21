from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import feddiffsecure.runtime_env  # noqa: F401
import numpy as np
import torch
from PIL import Image

from feddiffsecure.data import get_dataset
from feddiffsecure.diffusion import GaussianDiffusion, resolve_effective_sampler
from feddiffsecure.metrics import collect_real_images, compute_proxy_fid
from feddiffsecure.model import build_model
from feddiffsecure.semantic_metrics import compute_semantic_metrics
from feddiffsecure.utils import ensure_dir, load_yaml, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Systematic diffusion sampling diagnostics for one run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Trained run directory.")
    parser.add_argument(
        "--checkpoint-kinds",
        nargs="+",
        default=["final", "best_proxy", "best_semantic", "ema"],
        choices=["final", "best", "best_proxy", "best_semantic", "ema"],
        help="Checkpoints included in the sweep.",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["ddpm", "ddim"],
        choices=["ddpm", "ddim"],
        help="Sampling algorithms to compare.",
    )
    parser.add_argument(
        "--sample-steps",
        nargs="+",
        type=int,
        default=[50, 100, 200, 400, 600],
        help="Sampling step counts included in the sweep.",
    )
    parser.add_argument(
        "--ddim-etas",
        nargs="+",
        type=float,
        default=[0.0],
        help="DDIM eta values included in the sweep when sampler=ddim.",
    )
    parser.add_argument("--num-images", type=int, default=8, help="Generated samples per saved grid.")
    parser.add_argument("--num-real-samples", type=int, default=128, help="Real reference count for proxy FID.")
    parser.add_argument("--eval-num-workers", type=int, default=0, help="Workers for evaluation loaders.")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Optional override for eval batch size.")
    parser.add_argument("--image-scale", type=int, default=2, help="Nearest-neighbor upscaling factor for saved grids.")
    parser.add_argument("--sample-seed", type=int, default=123, help="Fixed seed used for every compared sample.")
    parser.add_argument("--output-dir", type=str, default="", help="Defaults to <run-dir>/sampling_diagnostics.")
    parser.add_argument("--disable-semantic-metrics", action="store_true", help="Skip classifier-based metrics.")
    parser.add_argument("--disable-clip-denoised", action="store_true", help="Disable x0 clipping.")
    parser.add_argument(
        "--trajectory-checkpoint",
        type=str,
        default="best_proxy",
        choices=["", "final", "best", "best_proxy", "best_semantic", "ema"],
        help="Optional checkpoint used for trajectory export.",
    )
    parser.add_argument(
        "--trajectory-sampler",
        type=str,
        default="ddim",
        choices=["ddpm", "ddim"],
        help="Sampler used for trajectory export.",
    )
    parser.add_argument("--trajectory-steps", type=int, default=200, help="Sampling steps used for trajectory export.")
    parser.add_argument(
        "--trajectory-capture-every",
        type=int,
        default=50,
        help="Reverse timesteps between saved trajectory frames.",
    )
    return parser.parse_args()


def _write_csv(records: list[dict], path: Path) -> None:
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


def _checkpoint_path(run_dir: Path, checkpoint_kind: str) -> tuple[Path, str]:
    path_map = {
        "final": run_dir / "global_model_final.pt",
        "best": run_dir / "global_model_best.pt",
        "best_proxy": run_dir / "global_model_best_proxy.pt",
        "best_semantic": run_dir / "global_model_best_semantic.pt",
        "ema": run_dir / "global_model_ema.pt",
    }
    checkpoint_path = path_map[checkpoint_kind]
    loaded_kind = checkpoint_kind
    if not checkpoint_path.exists():
        checkpoint_path = path_map["final"]
        loaded_kind = "final"
    return checkpoint_path, loaded_kind


def _save_tensor_grid(images: torch.Tensor, path: Path, nrow: int = 4, image_scale: int = 1) -> None:
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    batch, channels, height, width = images.shape
    nrow = max(1, min(nrow, batch))
    ncol = int(np.ceil(batch / nrow))
    canvas = torch.ones(channels, ncol * height, nrow * width)
    for idx in range(batch):
        row = idx // nrow
        col = idx % nrow
        canvas[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = images[idx]
    array = (canvas.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        image = Image.fromarray(array[:, :, 0], mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    if int(image_scale) > 1:
        image = image.resize((image.width * int(image_scale), image.height * int(image_scale)), Image.Resampling.NEAREST)
    image.save(path)


def _sanitize_name(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def _build_initial_noise(cfg: dict, num_images: int, sample_seed: int, device: str) -> torch.Tensor:
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(sample_seed))
    return torch.randn(
        (
            int(num_images),
            int(cfg["dataset"]["channels"]),
            int(cfg["dataset"]["image_size"]),
            int(cfg["dataset"]["image_size"]),
        ),
        generator=generator,
        device=generator_device,
    ).to(device)


def main() -> None:
    args = parse_args()
    set_seed(int(args.sample_seed))

    run_dir = Path(args.run_dir).resolve()
    output_dir = ensure_dir(args.output_dir or (run_dir / "sampling_diagnostics"))

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    cfg = load_yaml(run_summary["config_path"])
    device = resolve_device(str(cfg["system"]["device"]))
    eval_batch_size = int(args.eval_batch_size or cfg["evaluation"].get("eval_batch_size", 16))
    clip_denoised = False if args.disable_clip_denoised else bool(cfg["diffusion"].get("clip_denoised", True))

    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"].get("beta_schedule", "linear")),
    ).to(device)
    eval_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        train=False,
        image_size=int(cfg["dataset"]["image_size"]),
        channels=int(cfg["dataset"]["channels"]),
    )
    real_images = collect_real_images(
        eval_dataset,
        max_samples=int(args.num_real_samples),
        batch_size=eval_batch_size,
        num_workers=int(args.eval_num_workers),
    )
    initial_noise = _build_initial_noise(
        cfg=cfg,
        num_images=max(int(args.num_images), int(args.num_real_samples)),
        sample_seed=int(args.sample_seed),
        device=device,
    )

    records: list[dict] = []
    for checkpoint_kind in args.checkpoint_kinds:
        checkpoint_path, loaded_kind = _checkpoint_path(run_dir, str(checkpoint_kind))
        model = build_model(cfg["dataset"], cfg["model"]).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        for sampler in args.samplers:
            eta_values = [0.0] if sampler == "ddpm" else list(args.ddim_etas)
            for sample_steps in args.sample_steps:
                for ddim_eta in eta_values:
                    effective_sampler = resolve_effective_sampler(
                        sampler=sampler,
                        sample_steps=int(sample_steps),
                        timesteps=int(cfg["diffusion"]["timesteps"]),
                    )
                    fake_images = diffusion.sample(
                        model=model,
                        shape=tuple(initial_noise.shape),
                        device=device,
                        sample_steps=int(sample_steps),
                        sampler=str(sampler),
                        ddim_eta=float(ddim_eta),
                        clip_denoised=clip_denoised,
                        initial_noise=initial_noise,
                    ).cpu()

                    proxy_fid = compute_proxy_fid(
                        real_images=real_images[: fake_images.size(0)],
                        fake_images=fake_images,
                        feature_size=int(cfg["evaluation"].get("proxy_feature_size", 8)),
                    )
                    semantic_metrics = None
                    if not args.disable_semantic_metrics:
                        semantic_metrics = compute_semantic_metrics(
                            real_images=real_images[: fake_images.size(0)],
                            fake_images=fake_images,
                            dataset_name=str(cfg["dataset"]["name"]),
                            dataset_root=str(cfg["dataset"]["root"]),
                            image_size=int(cfg["dataset"]["image_size"]),
                            channels=int(cfg["dataset"]["channels"]),
                            device=device,
                            cache_dir=cfg["evaluation"].get("evaluator_cache_dir", "./outputs/evaluators"),
                            eval_batch_size=eval_batch_size,
                            num_workers=int(args.eval_num_workers),
                        )

                    tag = (
                        f"{loaded_kind}_{effective_sampler}_steps{int(sample_steps)}"
                        f"_eta{_sanitize_name(f'{float(ddim_eta):.2f}')}"
                    )
                    _save_tensor_grid(
                        fake_images[: int(args.num_images)],
                        output_dir / f"{tag}.png",
                        nrow=min(4, int(args.num_images)),
                        image_scale=int(args.image_scale),
                    )

                    record = {
                        "checkpoint_requested": str(checkpoint_kind),
                        "checkpoint_loaded": loaded_kind,
                        "sampler_requested": str(sampler),
                        "sampler_effective": effective_sampler,
                        "sample_steps": int(sample_steps),
                        "ddim_eta": float(ddim_eta),
                        "clip_denoised": bool(clip_denoised),
                        "beta_schedule": str(cfg["diffusion"].get("beta_schedule", "linear")),
                        "proxy_fid": float(proxy_fid),
                        "grid_path": str((output_dir / f"{tag}.png").resolve()),
                    }
                    if semantic_metrics is not None:
                        record.update(semantic_metrics)
                    records.append(record)

    trajectory_payload: dict[str, object] | None = None
    if args.trajectory_checkpoint:
        checkpoint_path, loaded_kind = _checkpoint_path(run_dir, str(args.trajectory_checkpoint))
        model = build_model(cfg["dataset"], cfg["model"]).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        trajectory_images, captures = diffusion.sample_with_trajectory(
            model=model,
            shape=tuple(initial_noise[: int(args.num_images)].shape),
            device=device,
            sample_steps=int(args.trajectory_steps),
            sampler=str(args.trajectory_sampler),
            ddim_eta=float(args.ddim_etas[0] if args.trajectory_sampler == "ddim" and args.ddim_etas else 0.0),
            clip_denoised=clip_denoised,
            initial_noise=initial_noise[: int(args.num_images)],
            capture_every=int(args.trajectory_capture_every),
            include_initial_noise=True,
        )
        trajectory_dir = ensure_dir(output_dir / "trajectory")
        capture_records: list[dict[str, object]] = []
        for idx, capture in enumerate(captures):
            timestep = int(capture["timestep"])
            capture_path = trajectory_dir / f"{idx:02d}_t{timestep:04d}.png"
            _save_tensor_grid(
                capture["samples"][: int(args.num_images)],
                capture_path,
                nrow=min(4, int(args.num_images)),
                image_scale=int(args.image_scale),
            )
            capture_records.append({"timestep": timestep, "grid_path": str(capture_path.resolve())})
        final_path = trajectory_dir / "final.png"
        _save_tensor_grid(
            trajectory_images[: int(args.num_images)],
            final_path,
            nrow=min(4, int(args.num_images)),
            image_scale=int(args.image_scale),
        )
        trajectory_payload = {
            "checkpoint_requested": str(args.trajectory_checkpoint),
            "checkpoint_loaded": loaded_kind,
            "sampler": str(args.trajectory_sampler),
            "sample_steps": int(args.trajectory_steps),
            "capture_every": int(args.trajectory_capture_every),
            "captures": capture_records,
            "final_grid_path": str(final_path.resolve()),
        }

    summary_payload = {
        "run_dir": str(run_dir),
        "sample_seed": int(args.sample_seed),
        "records": records,
        "trajectory": trajectory_payload,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(records, output_dir / "summary.csv")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
