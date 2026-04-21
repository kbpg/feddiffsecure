from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import feddiffsecure.runtime_env  # noqa: F401
import torch
from PIL import Image, ImageDraw, ImageFont

from feddiffsecure.data import get_dataset
from feddiffsecure.diffusion import GaussianDiffusion, resolve_effective_sampler
from feddiffsecure.metrics import collect_real_images, compute_proxy_fid
from feddiffsecure.model import build_model
from feddiffsecure.semantic_metrics import compute_semantic_metrics
from feddiffsecure.utils import ensure_dir, load_yaml, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sampling quality across multiple training runs.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Run output directories to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/model_comparison",
        help="Directory for comparison artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of generated samples per model for the qualitative grid.",
    )
    parser.add_argument(
        "--num-real-samples",
        type=int,
        default=256,
        help="Number of real samples used for proxy FID recomputation.",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=10,
        help="Number of trailing rounds used to summarize late-stage stability.",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="DataLoader workers used when recomputing comparison metrics.",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale factor for tiny images such as MNIST.",
    )
    parser.add_argument(
        "--sample-steps-override",
        type=int,
        default=0,
        help="Optional shared sample step count used for every run during recomputation and qualitative grids.",
    )
    parser.add_argument(
        "--report-title",
        type=str,
        default="Model sampling comparison",
        help="Title used in the generated markdown summary.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=["final", "best", "ema"],
        default="final",
        help="Which checkpoint to load from each run directory.",
    )
    parser.add_argument(
        "--sampler-override",
        type=str,
        choices=["ddpm", "ddim"],
        default="",
        help="Optional sampler override used for every compared run.",
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


def _tensor_to_pil(image: torch.Tensor, image_scale: int = 1) -> Image.Image:
    image = image.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1.0) / 2.0 * 255.0).byte()
    array = image.permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        pil_image = Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    else:
        pil_image = Image.fromarray(array, mode="RGB")

    if image_scale > 1:
        pil_image = pil_image.resize(
            (pil_image.width * image_scale, pil_image.height * image_scale),
            resample=Image.Resampling.NEAREST,
        )
    return pil_image


def _build_label_lines(record: dict) -> list[str]:
    sampler_requested = str(record.get("sampler_requested", record["sampler"]))
    sampler_effective = str(record["sampler"])
    sampler_label = sampler_effective if sampler_requested == sampler_effective else f"{sampler_requested}->{sampler_effective}"

    lines = [
        record["run_name"],
        (
            f"arch={record['architecture']}  base={record['base_channels']}  "
            f"rank={record['lora_rank']}  rounds={record['rounds']}  params={record['model_params_total']}"
        ),
        (
            f"best={record['best_proxy_fid']:.2f}  last={record['last_proxy_fid']:.2f}  "
            f"rebound={record['rebound']:.2f}  payload={record['payload_bytes_raw']}B"
        ),
        (
            f"lr={record['lr']:.0e}  local_steps={record['max_steps_per_epoch']}  "
            f"sample_steps={record['sampling_steps']}  sampler={sampler_label}  "
            f"{record['checkpoint_kind_loaded']}@eval={record['recomputed_eval_proxy_fid']:.2f}"
        ),
    ]
    if "classifier_fid" in record:
        lines.append(
            (
                f"classifier_fid={record['classifier_fid']:.2f}  "
                f"conf={record['top1_confidence_mean']:.3f}  "
                f"confident@0.9={record['top1_confident_ratio']:.3f}"
            )
        )
    return lines


def _make_sample_grid(
    images: torch.Tensor,
    label_lines: list[str],
    columns: int = 4,
    pad: int = 8,
    image_scale: int = 4,
) -> Image.Image:
    pil_images = [_tensor_to_pil(img, image_scale=image_scale) for img in images]
    if not pil_images:
        raise ValueError("At least one image is required to build a grid.")

    font = ImageFont.load_default()
    w, h = pil_images[0].size
    columns = max(1, min(columns, len(pil_images)))
    rows = math.ceil(len(pil_images) / columns)
    line_height = font.getbbox("Ag")[3] + 4
    header_h = pad + line_height * len(label_lines) + 4
    grid_w = pad + columns * (w + pad)
    grid_h = header_h + pad + rows * (h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), color="white")
    draw = ImageDraw.Draw(canvas)
    for idx, line in enumerate(label_lines):
        draw.text((pad, pad + idx * line_height), line, fill="black", font=font)

    for idx, img in enumerate(pil_images):
        row = idx // columns
        col = idx % columns
        x = pad + col * (w + pad)
        y = header_h + pad + row * (h + pad)
        canvas.paste(img, (x, y))

    return canvas


def _stack_grids(grids: list[Image.Image], pad: int = 12) -> Image.Image:
    width = max(grid.width for grid in grids)
    height = pad + sum(grid.height + pad for grid in grids)
    canvas = Image.new("RGB", (width, height), color="#f6f6f6")

    y = pad
    for grid in grids:
        canvas.paste(grid, (0, y))
        y += grid.height + pad
    return canvas


def _load_audit_payload_bytes(run_dir: Path) -> int | None:
    audit_path = run_dir / "audit_log.jsonl"
    if not audit_path.exists():
        return None
    first_line = audit_path.read_text(encoding="utf-8").splitlines()
    if not first_line:
        return None
    first_record = json.loads(first_line[0])
    return int(first_record["payload_bytes_raw"])


def _tail_stats(round_metrics: list[dict], tail_window: int) -> tuple[float, float]:
    tail = [record["proxy_fid"] for record in round_metrics[-min(tail_window, len(round_metrics)) :]]
    tail_arr = np.array(tail, dtype=np.float64)
    return float(tail_arr.mean()), float(tail_arr.std(ddof=0))


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_markdown_report(output_dir: Path, title: str, summary: dict) -> None:
    records = summary["runs"]
    has_semantic = bool(records) and "classifier_fid" in records[0]
    lines = [
        f"# {title}",
        "",
        f"![comparison_samples](comparison_samples.png)",
        "",
        "## Key findings",
        "",
        (
            f"- Best best-proxy-FID: `{summary['best_proxy_fid_run']['run_name']}` "
            f"({summary['best_proxy_fid_run']['best_proxy_fid']:.4f})"
        ),
        (
            f"- Best final-round proxy-FID: `{summary['best_final_proxy_fid_run']['run_name']}` "
            f"({summary['best_final_proxy_fid_run']['last_proxy_fid']:.4f})"
        ),
        (
            f"- Best late-stage stability: `{summary['best_stability_run']['run_name']}` "
            f"(rebound={summary['best_stability_run']['rebound']:.4f}, "
            f"tail_std={summary['best_stability_run']['tail_std_proxy_fid']:.4f})"
        ),
    ]
    if has_semantic and "best_classifier_fid_run" in summary:
        lines.append(
            (
                f"- Best semantic quality by classifier-FID: `{summary['best_classifier_fid_run']['run_name']}` "
                f"({summary['best_classifier_fid_run']['classifier_fid']:.4f})"
            )
        )

    lines.extend(["", "## Metrics", ""])
    if has_semantic:
        lines.extend(
            [
                "| run | checkpoint | rounds | lr | local_steps | sample_steps | best_proxy_fid | last_proxy_fid | proxy_eval_fid | classifier_fid | conf_mean | confident@0.9 | rebound | payload_bytes_raw |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| run | checkpoint | rounds | lr | local_steps | sample_steps | best_proxy_fid | last_proxy_fid | recomputed_eval_proxy_fid | rebound | payload_bytes_raw |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

    for record in records:
        if has_semantic:
            lines.append(
                "| "
                f"{record['run_name']} | "
                f"{record['checkpoint_kind_loaded']} | "
                f"{record['rounds']} | "
                f"{record['lr']:.0e} | "
                f"{record['max_steps_per_epoch']} | "
                f"{record['sampling_steps']} | "
                f"{record['best_proxy_fid']:.4f} | "
                f"{record['last_proxy_fid']:.4f} | "
                f"{record['recomputed_eval_proxy_fid']:.4f} | "
                f"{record['classifier_fid']:.4f} | "
                f"{record['top1_confidence_mean']:.4f} | "
                f"{record['top1_confident_ratio']:.4f} | "
                f"{record['rebound']:.4f} | "
                f"{record['payload_bytes_raw']} |"
            )
        else:
            lines.append(
                "| "
                f"{record['run_name']} | "
                f"{record['checkpoint_kind_loaded']} | "
                f"{record['rounds']} | "
                f"{record['lr']:.0e} | "
                f"{record['max_steps_per_epoch']} | "
                f"{record['sampling_steps']} | "
                f"{record['best_proxy_fid']:.4f} | "
                f"{record['last_proxy_fid']:.4f} | "
                f"{record['recomputed_eval_proxy_fid']:.4f} | "
                f"{record['rebound']:.4f} | "
                f"{record['payload_bytes_raw']} |"
            )

    lines.extend(
        [
            "",
            "## Qualitative panels",
            "",
            "The real reference row comes from the evaluation split. Each run row uses the same `sample_steps` value shown in the table, so the visual comparison and the recomputed proxy FID share the same sampling budget.",
            f"Checkpoint selection for this report: `{summary['checkpoint_kind_requested']}`.",
            "",
        ]
    )

    for record in records:
        lines.append(f"### {record['run_name']}")
        lines.append("")
        lines.append(f"![{record['run_name']}]({record['run_name']}_samples.png)")
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_seed(42)

    records: list[dict] = []
    grids: list[Image.Image] = []
    real_cache: dict[str, torch.Tensor] = {}
    noise_cache: dict[str, torch.Tensor] = {}

    for run_str in args.runs:
        run_dir = Path(run_str)
        run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        cfg = load_yaml(run_summary["config_path"])
        round_metrics = json.loads((run_dir / "metrics" / "round_metrics.json").read_text(encoding="utf-8"))
        final_metrics = json.loads((run_dir / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))
        sampling_steps = int(args.sample_steps_override) if int(args.sample_steps_override) > 0 else int(cfg["diffusion"]["sample_steps"])
        sampler = str(args.sampler_override or cfg["diffusion"].get("sampler", "ddpm"))
        ddim_eta = float(cfg["diffusion"].get("ddim_eta", 0.0) if args.ddim_eta_override < 0.0 else args.ddim_eta_override)
        clip_denoised = not bool(args.disable_clip_denoised) if args.disable_clip_denoised else bool(cfg["diffusion"].get("clip_denoised", True))
        diffusion_timesteps = int(cfg["diffusion"]["timesteps"])
        effective_sampler = resolve_effective_sampler(
            sampler=sampler,
            sample_steps=int(sampling_steps),
            timesteps=diffusion_timesteps,
        )

        device = resolve_device(str(cfg["system"]["device"]))
        model = build_model(cfg["dataset"], cfg["model"]).to(device)
        checkpoint_kind_loaded = str(args.checkpoint_kind)
        checkpoint_path = run_dir / f"global_model_{args.checkpoint_kind}.pt"
        if not checkpoint_path.exists():
            checkpoint_kind_loaded = "final"
            checkpoint_path = run_dir / "global_model_final.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        diffusion = GaussianDiffusion(timesteps=int(cfg["diffusion"]["timesteps"])).to(device)

        dataset_key = json.dumps(
            {
                "name": cfg["dataset"]["name"],
                "root": cfg["dataset"]["root"],
                "image_size": cfg["dataset"]["image_size"],
                "channels": cfg["dataset"]["channels"],
                "num_real_samples": args.num_real_samples,
                "eval_batch_size": cfg["evaluation"]["eval_batch_size"],
                "num_workers": cfg["training"].get("num_workers", 0),
            },
            sort_keys=True,
        )
        if dataset_key not in real_cache:
            eval_dataset = get_dataset(
                name=cfg["dataset"]["name"],
                root=cfg["dataset"]["root"],
                train=False,
                image_size=int(cfg["dataset"]["image_size"]),
                channels=int(cfg["dataset"]["channels"]),
            )
            real_cache[dataset_key] = collect_real_images(
                eval_dataset,
                max_samples=int(args.num_real_samples),
                batch_size=int(cfg["evaluation"]["eval_batch_size"]),
                num_workers=int(args.eval_num_workers),
            )
            noise_cache[dataset_key] = torch.randn(
                (
                    int(args.num_real_samples),
                    int(cfg["dataset"]["channels"]),
                    int(cfg["dataset"]["image_size"]),
                    int(cfg["dataset"]["image_size"]),
                )
            )

        if not any(grid.info.get("kind") == f"real::{cfg['dataset']['name']}" for grid in grids):
            real_record = {
                "run_name": f"{cfg['dataset']['name']}_real_reference",
                "architecture": "real_data",
                "base_channels": 0,
                "lora_rank": 0,
                "model_params_total": 0,
                "best_proxy_fid": 0.0,
                "last_proxy_fid": 0.0,
                "rebound": 0.0,
                "payload_bytes_raw": 0,
            }
            real_grid = _make_sample_grid(
                real_cache[dataset_key][: int(args.num_samples)],
                label_lines=[
                    real_record["run_name"],
                    "reference images from the evaluation split",
                    "used for qualitative comparison only",
                ],
                image_scale=int(args.image_scale),
            )
            real_grid.info["kind"] = f"real::{cfg['dataset']['name']}"
            grids.append(real_grid)
            real_grid.save(output_dir / f"{cfg['dataset']['name']}_real_reference.png")

        with torch.no_grad():
            fake_images = diffusion.sample(
                model,
                shape=tuple(noise_cache[dataset_key].shape),
                device=device,
                sample_steps=sampling_steps,
                sampler=sampler,
                ddim_eta=ddim_eta,
                clip_denoised=clip_denoised,
                initial_noise=noise_cache[dataset_key],
            ).cpu()

        recomputed_proxy_fid = compute_proxy_fid(
            real_images=real_cache[dataset_key][: fake_images.size(0)],
            fake_images=fake_images,
            feature_size=int(cfg["evaluation"]["proxy_feature_size"]),
        )
        semantic_metrics = None
        if not bool(args.disable_semantic_metrics):
            semantic_metrics = compute_semantic_metrics(
                real_images=real_cache[dataset_key][: fake_images.size(0)],
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

        best_record = final_metrics["best_proxy_fid_round"]
        last_record = final_metrics["last_round"]
        tail_mean, tail_std = _tail_stats(round_metrics, args.tail_window)
        payload_bytes = _load_audit_payload_bytes(run_dir)

        run_name = run_dir.name
        model_cfg = cfg["model"]
        record = {
            "run_name": run_name,
            "dataset": cfg["dataset"]["name"],
            "architecture": str(model_cfg.get("architecture", "tiny_unet")),
            "base_channels": int(model_cfg["base_channels"]),
            "channel_multipliers": "-".join(str(x) for x in model_cfg.get("channel_multipliers", [])),
            "blocks_per_level": int(model_cfg.get("blocks_per_level", 1 if model_cfg.get("architecture", "tiny_unet") == "tiny_unet" else 2)),
            "lora_rank": int(model_cfg["lora_rank"]),
            "lora_alpha": float(model_cfg["lora_alpha"]),
            "model_params_total": int(run_summary["model_params_total"]),
            "model_params_trainable": int(run_summary["model_params_trainable"]),
            "rounds": int(run_summary["rounds"]),
            "lr": float(run_summary["lr"]),
            "max_steps_per_epoch": int(run_summary["max_steps_per_epoch"]),
            "clients_per_round": int(run_summary["clients_per_round"]),
            "sampling_steps": sampling_steps,
            "sampler": effective_sampler,
            "sampler_requested": sampler,
            "sampler_effective": effective_sampler,
            "ddim_eta": ddim_eta,
            "clip_denoised": bool(clip_denoised),
            "checkpoint_kind_requested": str(args.checkpoint_kind),
            "checkpoint_kind_loaded": checkpoint_kind_loaded,
            "best_round": int(best_record["round"]),
            "best_proxy_fid": float(best_record["proxy_fid"]),
            "last_proxy_fid": float(last_record["proxy_fid"]),
            "recomputed_eval_proxy_fid": float(recomputed_proxy_fid),
            "rebound": float(last_record["proxy_fid"]) - float(best_record["proxy_fid"]),
            "tail_mean_proxy_fid": tail_mean,
            "tail_std_proxy_fid": tail_std,
            "last_avg_client_loss": float(last_record["avg_client_loss"]),
            "payload_bytes_raw": payload_bytes if payload_bytes is not None else 0,
        }
        if semantic_metrics is not None:
            record.update(semantic_metrics)
        sample_grid = _make_sample_grid(
            fake_images[: int(args.num_samples)],
            label_lines=_build_label_lines(record),
            image_scale=int(args.image_scale),
        )
        grids.append(sample_grid)
        sample_grid.save(output_dir / f"{run_name}_samples.png")

        run_samples_dir = ensure_dir(run_dir / "samples")
        sample_grid.save(run_samples_dir / f"{checkpoint_kind_loaded}_compare.png")

        records.append(record)

    def _sort_key(item: dict) -> tuple:
        if "classifier_fid" in item:
            return (item["classifier_fid"], item["recomputed_eval_proxy_fid"], item["last_proxy_fid"])
        return (item["best_proxy_fid"], item["last_proxy_fid"], item["rebound"])

    records.sort(key=_sort_key)

    summary = {
        "checkpoint_kind_requested": str(args.checkpoint_kind),
        "runs": records,
        "best_proxy_fid_run": min(records, key=lambda item: item["best_proxy_fid"]),
        "best_final_proxy_fid_run": min(records, key=lambda item: item["last_proxy_fid"]),
        "best_stability_run": min(records, key=lambda item: (item["rebound"], item["tail_std_proxy_fid"], item["last_proxy_fid"])),
    }
    if records and "classifier_fid" in records[0]:
        summary["best_classifier_fid_run"] = min(records, key=lambda item: item["classifier_fid"])
        summary["best_confidence_run"] = max(records, key=lambda item: item["top1_confidence_mean"])

    comparison_grid = _stack_grids(grids)
    comparison_grid.save(output_dir / "comparison_samples.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(records, output_dir / "summary.csv")
    _write_markdown_report(output_dir, args.report_title, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
