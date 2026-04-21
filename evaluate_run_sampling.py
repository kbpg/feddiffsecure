from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import feddiffsecure.runtime_env  # noqa: F401
import torch

from compare_runs import _make_sample_grid, _stack_grids
from feddiffsecure.data import get_dataset
from feddiffsecure.diffusion import GaussianDiffusion, resolve_effective_sampler
from feddiffsecure.metrics import collect_real_images, compute_proxy_fid
from feddiffsecure.model import build_model
from feddiffsecure.semantic_metrics import compute_semantic_metrics
from feddiffsecure.utils import ensure_dir, load_yaml, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one run at multiple diffusion sampling step counts.")
    parser.add_argument("--run-dir", type=str, required=True, help="Trained run directory.")
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="+",
        required=True,
        help="Sampling steps to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to <run-dir>/sampling_eval.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Generated images per qualitative grid.")
    parser.add_argument("--num-real-samples", type=int, default=256, help="Reference sample count for proxy FID.")
    parser.add_argument("--eval-num-workers", type=int, default=0, help="Workers for evaluation loaders.")
    parser.add_argument("--image-scale", type=int, default=6, help="Nearest-neighbor upscale factor.")
    parser.add_argument(
        "--sampler-override",
        type=str,
        choices=["ddpm", "ddim"],
        default="",
        help="Optional sampler override used for every sampling step count.",
    )
    parser.add_argument(
        "--ddim-eta-override",
        type=float,
        default=-1.0,
        help="Optional DDIM eta override. Negative values keep the config default.",
    )
    parser.add_argument(
        "--disable-clip-denoised",
        action="store_true",
        help="Disable x0 clipping during sampling.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=["final", "best", "best_proxy", "best_semantic", "ema"],
        default="final",
        help="Which checkpoint to evaluate.",
    )
    parser.add_argument(
        "--disable-semantic-metrics",
        action="store_true",
        help="Skip classifier-based semantic metrics.",
    )
    parser.add_argument(
        "--evaluator-cache-dir",
        type=str,
        default="./outputs/evaluators",
        help="Directory used to cache the MNIST/Fashion-MNIST evaluator.",
    )
    return parser.parse_args()


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    set_seed(42)

    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(args.output_dir or (run_dir / "sampling_eval"))

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    cfg = load_yaml(run_summary["config_path"])

    device = resolve_device(str(cfg["system"]["device"]))
    model = build_model(cfg["dataset"], cfg["model"]).to(device)
    checkpoint_name_map = {
        "final": "global_model_final.pt",
        "best": "global_model_best.pt",
        "best_proxy": "global_model_best_proxy.pt",
        "best_semantic": "global_model_best_semantic.pt",
        "ema": "global_model_ema.pt",
    }
    checkpoint_path = run_dir / checkpoint_name_map[str(args.checkpoint_kind)]
    checkpoint_kind_loaded = str(args.checkpoint_kind)
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "global_model_final.pt"
        checkpoint_kind_loaded = "final"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"].get("beta_schedule", "linear")),
    ).to(device)
    sampler = str(args.sampler_override or cfg["diffusion"].get("sampler", "ddpm"))
    ddim_eta = float(cfg["diffusion"].get("ddim_eta", 0.0) if args.ddim_eta_override < 0.0 else args.ddim_eta_override)
    clip_denoised = not bool(args.disable_clip_denoised) if args.disable_clip_denoised else bool(cfg["diffusion"].get("clip_denoised", True))
    diffusion_timesteps = int(cfg["diffusion"]["timesteps"])

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
        batch_size=int(cfg["evaluation"]["eval_batch_size"]),
        num_workers=int(args.eval_num_workers),
    )

    grids = [
        _make_sample_grid(
            real_images[: int(args.num_samples)],
            label_lines=[
                f"{cfg['dataset']['name']}_real_reference",
                "reference images from the evaluation split",
                "qualitative target for the generated samples",
            ],
            image_scale=int(args.image_scale),
        )
    ]
    records: list[dict] = []
    initial_noise = torch.randn(
        (
            int(args.num_real_samples),
            int(cfg["dataset"]["channels"]),
            int(cfg["dataset"]["image_size"]),
            int(cfg["dataset"]["image_size"]),
        )
    )

    for sample_steps in args.sample_steps:
        effective_sampler = resolve_effective_sampler(
            sampler=sampler,
            sample_steps=int(sample_steps),
            timesteps=diffusion_timesteps,
        )
        with torch.no_grad():
            fake_images = diffusion.sample(
                model,
                shape=tuple(initial_noise.shape),
                device=device,
                sample_steps=int(sample_steps),
                sampler=sampler,
                ddim_eta=ddim_eta,
                clip_denoised=clip_denoised,
                initial_noise=initial_noise,
            ).cpu()

        proxy_fid = compute_proxy_fid(
            real_images=real_images[: fake_images.size(0)],
            fake_images=fake_images,
            feature_size=int(cfg["evaluation"]["proxy_feature_size"]),
        )

        semantic_metrics = None
        if not bool(args.disable_semantic_metrics):
            semantic_metrics = compute_semantic_metrics(
                real_images=real_images[: fake_images.size(0)],
                fake_images=fake_images,
                dataset_name=str(cfg["dataset"]["name"]),
                dataset_root=str(cfg["dataset"]["root"]),
                image_size=int(cfg["dataset"]["image_size"]),
                channels=int(cfg["dataset"]["channels"]),
                device=device,
                cache_dir=args.evaluator_cache_dir,
                eval_batch_size=int(cfg["evaluation"]["eval_batch_size"]),
                num_workers=int(args.eval_num_workers),
            )

        record = {
            "sample_steps": int(sample_steps),
            "proxy_fid": float(proxy_fid),
            "sampler": effective_sampler,
            "sampler_requested": sampler,
            "sampler_effective": effective_sampler,
            "ddim_eta": float(ddim_eta),
            "clip_denoised": bool(clip_denoised),
            "checkpoint_kind_loaded": checkpoint_kind_loaded,
        }
        if semantic_metrics is not None:
            record.update(semantic_metrics)
        records.append(record)

        label_lines = [
            f"{run_dir.name} | checkpoint={checkpoint_kind_loaded} | sample_steps={sample_steps}",
            f"proxy_fid={proxy_fid:.2f}",
            (
                f"sampler={effective_sampler}  requested={sampler}  eta={ddim_eta:.2f}  clip={clip_denoised}  "
                f"arch={cfg['model'].get('architecture', 'tiny_unet')}  base={cfg['model']['base_channels']}  rank={cfg['model']['lora_rank']}"
            ),
        ]
        if semantic_metrics is not None:
            label_lines.append(
                (
                    f"classifier_fid={semantic_metrics['classifier_fid']:.2f}  "
                    f"conf={semantic_metrics['top1_confidence_mean']:.3f}  "
                    f"confident@0.9={semantic_metrics['top1_confident_ratio']:.3f}"
                )
            )

        grid = _make_sample_grid(
            fake_images[: int(args.num_samples)],
            label_lines=label_lines,
            image_scale=int(args.image_scale),
        )
        grids.append(grid)
        grid.save(output_dir / f"samples_steps_{sample_steps}.png")

    _stack_grids(grids).save(output_dir / "sampling_steps_comparison.png")
    (output_dir / "summary.json").write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(records, output_dir / "summary.csv")

    print(json.dumps({"records": records}, indent=2, ensure_ascii=False))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
