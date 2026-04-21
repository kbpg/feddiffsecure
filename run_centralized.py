from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import feddiffsecure.runtime_env  # noqa: F401
import torch
from torch.optim import AdamW

from feddiffsecure.data import build_loader, get_dataset
from feddiffsecure.diffusion import GaussianDiffusion
from feddiffsecure.lora import enable_parameter_prefixes, trainable_parameter_names
from feddiffsecure.metrics import collect_real_images, compute_proxy_fid
from feddiffsecure.model import build_model
from feddiffsecure.semantic_metrics import compute_semantic_metrics, supports_semantic_metrics
from feddiffsecure.server import FederatedServer
from feddiffsecure.utils import (
    count_parameters,
    ensure_dir,
    load_yaml,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized diffusion training baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any incomplete checkpoint in the configured output directory and start fresh.",
    )
    return parser.parse_args()


def _write_round_metrics_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _update_ema_state(
    ema_state: dict[str, torch.Tensor],
    module: torch.nn.Module,
    decay: float,
) -> None:
    for name, tensor in module.state_dict().items():
        current = tensor.detach()
        if name in ema_state and ema_state[name].device != current.device:
            ema_state[name] = ema_state[name].to(current.device)
        if not current.is_floating_point():
            ema_state[name] = current.clone()
            continue
        ema_state[name].mul_(decay).add_(current, alpha=1.0 - decay)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_training_state(path: Path, state: dict) -> None:
    ema_state = state.get("ema_state")
    if ema_state is not None:
        state = dict(state)
        state["ema_state"] = {name: tensor.detach().cpu() for name, tensor in ema_state.items()}
    torch.save(state, path)


def _move_state_tensors(state: dict[str, torch.Tensor] | None, device: str) -> dict[str, torch.Tensor] | None:
    if state is None:
        return None
    return {name: tensor.to(device) for name, tensor in state.items()}


def _build_fixed_sample_noise(cfg: dict, device: str, num_images: int, sample_seed: int) -> torch.Tensor:
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
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    run_started_at = datetime.now().astimezone().isoformat()
    run_start_time = time.perf_counter()

    device = resolve_device(str(cfg["system"]["device"]))
    output_dir = ensure_dir(cfg["system"]["output_dir"])
    samples_dir = ensure_dir(output_dir / "samples")
    metrics_dir = ensure_dir(output_dir / "metrics")
    summary_path = output_dir / "run_summary.json"
    final_metrics_path = metrics_dir / "final_metrics.json"
    latest_state_path = output_dir / "training_state.pt"
    latest_model_path = output_dir / "global_model_latest.pt"

    if final_metrics_path.exists() and not args.restart:
        print(f"Run already completed at {output_dir}; skipping because final_metrics.json already exists.")
        return

    train_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        train=True,
        image_size=int(cfg["dataset"]["image_size"]),
        channels=int(cfg["dataset"]["channels"]),
    )
    eval_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        train=False,
        image_size=int(cfg["dataset"]["image_size"]),
        channels=int(cfg["dataset"]["channels"]),
    )
    save_json(
        {
            "train_examples": int(len(train_dataset)),
            "eval_examples": int(len(eval_dataset)),
        },
        output_dir / "dataset_summary.json",
    )

    model = build_model(cfg["dataset"], cfg["model"])
    unfreeze_prefixes = [str(x) for x in cfg.get("training", {}).get("unfreeze_param_prefixes", [])]
    enabled_param_names = enable_parameter_prefixes(model, unfreeze_prefixes)
    trainable_names = trainable_parameter_names(model)
    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"].get("beta_schedule", "linear")),
    ).to(device)
    server = FederatedServer(model=model, diffusion=diffusion, cfg=cfg, device=device)

    central_cfg = cfg.get("centralized", {})
    rounds = int(central_cfg.get("rounds", cfg.get("federated", {}).get("rounds", 1)))
    max_steps_per_round = int(
        central_cfg.get("max_steps_per_round", cfg.get("federated", {}).get("max_steps_per_epoch", 0))
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=True,
    )
    effective_max_steps = max_steps_per_round if max_steps_per_round > 0 else len(train_loader)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    x0_loss_weight = float(cfg.get("diffusion", {}).get("x0_loss_weight", 0.0))
    ema_decay = float(cfg["training"].get("ema_decay", 0.0))
    ema_state: dict[str, torch.Tensor] | None = None

    round_metrics: list[dict] = []
    round_durations_sec: list[float] = []
    best_proxy_fid = float("inf")
    best_proxy_round_id = 0
    best_semantic_fid = float("inf")
    best_semantic_round_id = 0
    best_checkpoint_metric = "proxy_fid"
    start_round = 1
    elapsed_wall_clock_before_resume = 0.0
    resume_count = 0

    existing_summary = _load_json(summary_path)
    if latest_state_path.exists() and not args.restart:
        resume_state = torch.load(latest_state_path, map_location="cpu")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        ema_state = _move_state_tensors(resume_state.get("ema_state"), device)
        round_metrics = list(resume_state.get("round_metrics", []))
        round_durations_sec = [float(x) for x in resume_state.get("round_durations_sec", [])]
        best_proxy_fid = float(resume_state.get("best_proxy_fid", best_proxy_fid))
        best_proxy_round_id = int(resume_state.get("best_proxy_round_id", best_proxy_round_id))
        best_semantic_fid = float(resume_state.get("best_semantic_fid", best_semantic_fid))
        best_semantic_round_id = int(resume_state.get("best_semantic_round_id", best_semantic_round_id))
        best_checkpoint_metric = str(resume_state.get("best_checkpoint_metric", best_checkpoint_metric))
        start_round = int(resume_state.get("completed_round", len(round_metrics))) + 1
        elapsed_wall_clock_before_resume = float(resume_state.get("elapsed_wall_clock_seconds", 0.0))
        run_started_at = str(resume_state.get("started_at", run_started_at))
        resume_count = int(resume_state.get("resume_count", 0)) + 1
    elif existing_summary is not None:
        resume_count = int(existing_summary.get("resume_count", 0))

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "started_at": run_started_at,
        "training_mode": "centralized",
        "device": device,
        "dataset": cfg["dataset"]["name"],
        "model_params_total": count_parameters(model),
        "model_params_trainable": count_parameters(model, only_trainable=True),
        "num_clients": 1,
        "rounds": rounds,
        "clients_per_round": 1,
        "local_epochs": 1,
        "max_steps_per_epoch": effective_max_steps,
        "dirichlet_alpha": 0.0,
        "lr": float(cfg["training"]["lr"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "ema_decay": ema_decay,
        "x0_loss_weight": x0_loss_weight,
        "semantic_eval_every": int(cfg.get("evaluation", {}).get("semantic_eval_every", 0)),
        "beta_schedule": str(cfg["diffusion"].get("beta_schedule", "linear")),
        "trainable_param_tensors": len(trainable_names),
        "unfreeze_param_prefixes": unfreeze_prefixes,
        "enabled_extra_params": len(enabled_param_names),
        "train_examples": int(len(train_dataset)),
        "resume_count": int(resume_count),
    }
    if start_round > 1:
        summary["resumed_from_round"] = int(start_round - 1)
        summary["last_resumed_at"] = datetime.now().astimezone().isoformat()
        summary["elapsed_wall_clock_seconds_before_resume"] = round(float(elapsed_wall_clock_before_resume), 3)
    save_json(summary, output_dir / "run_summary.json")

    print("=" * 80)
    print("Centralized diffusion baseline")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)

    eval_cfg = cfg.get("evaluation", {})
    semantic_eval_every = int(eval_cfg.get("semantic_eval_every", 0))
    semantic_eval_enabled = semantic_eval_every > 0 and supports_semantic_metrics(str(cfg["dataset"]["name"]))
    real_eval_images = collect_real_images(
        eval_dataset,
        max_samples=int(eval_cfg.get("num_real_samples", 128)),
        batch_size=int(eval_cfg.get("eval_batch_size", 128)),
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )
    eval_fake_count = int(eval_cfg.get("num_fake_samples", 128))
    eval_sample_seed = int(eval_cfg.get("eval_sample_seed", cfg["seed"]))
    eval_initial_noise = _build_fixed_sample_noise(
        cfg=cfg,
        device=device,
        num_images=eval_fake_count,
        sample_seed=eval_sample_seed,
    )
    summary["eval_sample_seed"] = int(eval_sample_seed)
    save_json(summary, output_dir / "run_summary.json")

    for round_id in range(start_round, rounds + 1):
        round_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_steps = 0
        total_samples_seen = 0

        for step, (x, _) in enumerate(train_loader):
            if max_steps_per_round > 0 and step >= max_steps_per_round:
                break
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.p_losses(model, x, t, x0_loss_weight=x0_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema_decay > 0.0:
                if ema_state is None:
                    ema_state = _clone_state_dict(model)
                else:
                    _update_ema_state(ema_state, model, decay=ema_decay)
            total_loss += float(loss.item())
            total_steps += 1
            total_samples_seen += int(x.size(0))

        if total_steps == 0:
            raise RuntimeError("No training steps were executed. Check max_steps_per_round and DataLoader settings.")

        fake_images = server.generate_samples(
            num_images=eval_fake_count,
            initial_noise=eval_initial_noise,
        ).cpu()
        proxy_fid = compute_proxy_fid(
            real_images=real_eval_images[: fake_images.size(0)],
            fake_images=fake_images,
            feature_size=int(eval_cfg.get("proxy_feature_size", 8)),
        )

        semantic_metrics = None
        if semantic_eval_enabled and round_id % semantic_eval_every == 0:
            semantic_metrics = compute_semantic_metrics(
                real_images=real_eval_images[: fake_images.size(0)],
                fake_images=fake_images,
                dataset_name=str(cfg["dataset"]["name"]),
                dataset_root=str(cfg["dataset"]["root"]),
                image_size=int(cfg["dataset"]["image_size"]),
                channels=int(cfg["dataset"]["channels"]),
                device=device,
                cache_dir=eval_cfg.get("evaluator_cache_dir", "./outputs/evaluators"),
                eval_batch_size=int(eval_cfg.get("eval_batch_size", 128)),
                num_workers=int(cfg["training"].get("num_workers", 0)),
            )

        if bool(cfg["system"].get("save_every_round", False)):
            server.save_samples(str(samples_dir / f"round_{round_id:03d}.png"), num_images=4)

        round_duration_sec = time.perf_counter() - round_start_time
        round_record = {
            "round": round_id,
            "avg_client_loss": round(total_loss / max(total_steps, 1), 6),
            "proxy_fid": round(float(proxy_fid), 6),
            "total_samples": int(total_samples_seen),
            "aggregated_clients": 1,
            "round_duration_sec": round(float(round_duration_sec), 3),
        }
        if semantic_metrics is not None:
            round_record.update(semantic_metrics)
        round_metrics.append(round_record)
        round_durations_sec.append(round_duration_sec)

        if float(proxy_fid) < best_proxy_fid:
            best_proxy_fid = float(proxy_fid)
            best_proxy_round_id = round_id
            torch.save(model.state_dict(), output_dir / "global_model_best_proxy.pt")
            if best_semantic_round_id == 0:
                torch.save(model.state_dict(), output_dir / "global_model_best.pt")

        if semantic_metrics is not None:
            classifier_fid = float(semantic_metrics["classifier_fid"])
            if classifier_fid < best_semantic_fid:
                best_semantic_fid = classifier_fid
                best_semantic_round_id = round_id
                best_checkpoint_metric = "classifier_fid"
                torch.save(model.state_dict(), output_dir / "global_model_best_semantic.pt")
                torch.save(model.state_dict(), output_dir / "global_model_best.pt")

        save_json(round_metrics, metrics_dir / "round_metrics.json")
        _write_round_metrics_csv(round_metrics, metrics_dir / "round_metrics.csv")
        torch.save(model.state_dict(), latest_model_path)
        elapsed_wall_clock_so_far = elapsed_wall_clock_before_resume + (time.perf_counter() - run_start_time) + sum(
            round_durations_sec[:-1]
        )
        _save_training_state(
            latest_state_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state": ema_state,
                "round_metrics": round_metrics,
                "round_durations_sec": round_durations_sec,
                "best_proxy_fid": best_proxy_fid,
                "best_proxy_round_id": best_proxy_round_id,
                "best_semantic_fid": best_semantic_fid,
                "best_semantic_round_id": best_semantic_round_id,
                "best_checkpoint_metric": best_checkpoint_metric,
                "completed_round": round_id,
                "started_at": run_started_at,
                "elapsed_wall_clock_seconds": float(elapsed_wall_clock_before_resume + sum(round_durations_sec)),
                "resume_count": int(resume_count),
            },
        )

        summary_line = (
            f"[Round {round_id}] avg_loss={total_loss / max(total_steps, 1):.4f}, "
            f"proxy_fid={proxy_fid:.4f}, total_samples={total_samples_seen}, "
            f"round_sec={round_duration_sec:.1f}"
        )
        if semantic_metrics is not None:
            summary_line += (
                f", classifier_fid={semantic_metrics['classifier_fid']:.2f}, "
                f"conf={semantic_metrics['top1_confidence_mean']:.3f}"
            )
        print(summary_line)

    torch.save(model.state_dict(), output_dir / "global_model_final.pt")
    if ema_state is not None:
        torch.save(ema_state, output_dir / "global_model_ema.pt")

    total_wall_clock_sec = elapsed_wall_clock_before_resume + (time.perf_counter() - run_start_time)
    final_metrics = {
        "last_round": round_metrics[-1],
        "best_proxy_fid_round": min(round_metrics, key=lambda x: x["proxy_fid"]),
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_path": str((output_dir / "global_model_best.pt").resolve()),
        "best_proxy_checkpoint_path": str((output_dir / "global_model_best_proxy.pt").resolve()),
        "wall_clock_seconds": round(float(total_wall_clock_sec), 3),
        "avg_round_seconds": round(float(sum(round_durations_sec) / max(len(round_durations_sec), 1)), 3),
        "max_round_seconds": round(float(max(round_durations_sec)), 3),
    }
    semantic_rounds = [record for record in round_metrics if "classifier_fid" in record]
    if semantic_rounds:
        final_metrics["best_classifier_fid_round"] = min(semantic_rounds, key=lambda x: x["classifier_fid"])
        final_metrics["best_semantic_checkpoint_path"] = str((output_dir / "global_model_best_semantic.pt").resolve())
    if ema_state is not None:
        final_metrics["ema_checkpoint_path"] = str((output_dir / "global_model_ema.pt").resolve())
    save_json(final_metrics, metrics_dir / "final_metrics.json")

    summary["completed_at"] = datetime.now().astimezone().isoformat()
    summary["wall_clock_seconds"] = round(float(total_wall_clock_sec), 3)
    if round_durations_sec:
        summary["avg_round_seconds"] = round(float(sum(round_durations_sec) / len(round_durations_sec)), 3)
        summary["max_round_seconds"] = round(float(max(round_durations_sec)), 3)
    save_json(summary, output_dir / "run_summary.json")

    if bool(cfg["system"].get("save_final_samples", False)):
        fixed_num_images = int(cfg["system"].get("final_sample_num_images", 4))
        fixed_sample_seed = int(cfg["system"].get("sample_seed", cfg["seed"]))
        shared_noise = _build_fixed_sample_noise(
            cfg=cfg,
            device=device,
            num_images=fixed_num_images,
            sample_seed=fixed_sample_seed,
        )
        summary["sample_seed"] = int(fixed_sample_seed)
        summary["final_sample_num_images"] = int(fixed_num_images)
        save_json(summary, output_dir / "run_summary.json")

        server.save_samples(
            str(samples_dir / "final.png"),
            num_images=fixed_num_images,
            initial_noise=shared_noise,
        )
        if (best_semantic_round_id > 0 or best_proxy_round_id > 0) and (output_dir / "global_model_best.pt").exists():
            original_model = server.model
            best_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            best_model.load_state_dict(torch.load(output_dir / "global_model_best.pt", map_location=device))
            server.model = best_model
            server.save_samples(
                str(samples_dir / "best.png"),
                num_images=fixed_num_images,
                initial_noise=shared_noise,
            )
            server.model = original_model
        if best_proxy_round_id > 0 and (output_dir / "global_model_best_proxy.pt").exists():
            original_model = server.model
            best_proxy_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            best_proxy_model.load_state_dict(torch.load(output_dir / "global_model_best_proxy.pt", map_location=device))
            server.model = best_proxy_model
            server.save_samples(
                str(samples_dir / "best_proxy.png"),
                num_images=fixed_num_images,
                initial_noise=shared_noise,
            )
            server.model = original_model
        if best_semantic_round_id > 0 and (output_dir / "global_model_best_semantic.pt").exists():
            original_model = server.model
            best_semantic_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            best_semantic_model.load_state_dict(
                torch.load(output_dir / "global_model_best_semantic.pt", map_location=device)
            )
            server.model = best_semantic_model
            server.save_samples(
                str(samples_dir / "best_semantic.png"),
                num_images=fixed_num_images,
                initial_noise=shared_noise,
            )
            server.model = original_model
        if ema_state is not None and (output_dir / "global_model_ema.pt").exists():
            original_model = server.model
            ema_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            ema_model.load_state_dict(torch.load(output_dir / "global_model_ema.pt", map_location=device))
            server.model = ema_model
            server.save_samples(
                str(samples_dir / "ema.png"),
                num_images=fixed_num_images,
                initial_noise=shared_noise,
            )
            server.model = original_model

    print("\nDone.")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
