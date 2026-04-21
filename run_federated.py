from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
import csv

import feddiffsecure.runtime_env  # noqa: F401
import torch

from feddiffsecure.audit import AuditLogger, summarize_audit_log
from feddiffsecure.client import FederatedClient
from feddiffsecure.data import dirichlet_split, get_dataset, summarize_client_splits
from feddiffsecure.diffusion import GaussianDiffusion
from feddiffsecure.lora import enable_parameter_prefixes, trainable_parameter_names
from feddiffsecure.metrics import collect_real_images, compute_proxy_fid
from feddiffsecure.model import build_model
from feddiffsecure.semantic_metrics import compute_semantic_metrics, supports_semantic_metrics
from feddiffsecure.server import FederatedServer
from feddiffsecure.stego import package_state_as_stego_png, recover_state_from_stego_png
from feddiffsecure.utils import (
    count_parameters,
    ensure_dir,
    load_yaml,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal FedDiffSecure prototype")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to YAML config")
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
        if not current.is_floating_point():
            ema_state[name] = current.clone()
            continue
        ema_state[name].mul_(decay).add_(current, alpha=1.0 - decay)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    run_started_at = datetime.now().astimezone().isoformat()
    run_start_time = time.perf_counter()

    device = resolve_device(str(cfg["system"]["device"]))
    output_dir = ensure_dir(cfg["system"]["output_dir"])
    carriers_dir = ensure_dir(output_dir / "carriers")
    samples_dir = ensure_dir(output_dir / "samples")
    metrics_dir = ensure_dir(output_dir / "metrics")

    # -------------------------
    # 数据集
    # -------------------------
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

    client_subsets = dirichlet_split(
        train_dataset,
        num_clients=int(cfg["federated"]["num_clients"]),
        alpha=float(cfg["federated"]["dirichlet_alpha"]),
        seed=int(cfg["seed"]),
    )

    dataset_summary = summarize_client_splits(client_subsets)
    save_json({"clients": dataset_summary}, output_dir / "client_splits.json")

    # -------------------------
    # 模型 / 服务端
    # -------------------------
    model = build_model(cfg["dataset"], cfg["model"])
    unfreeze_prefixes = [str(x) for x in cfg.get("training", {}).get("unfreeze_param_prefixes", [])]
    enabled_param_names = enable_parameter_prefixes(model, unfreeze_prefixes)
    trainable_names = trainable_parameter_names(model)
    diffusion = GaussianDiffusion(timesteps=int(cfg["diffusion"]["timesteps"])).to(device)
    server = FederatedServer(model=model, diffusion=diffusion, cfg=cfg, device=device)
    audit_logger = AuditLogger(output_dir / "audit_log.jsonl")
    save_carriers = bool(cfg.get("stego", {}).get("save_carriers", True))

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "started_at": run_started_at,
        "device": device,
        "dataset": cfg["dataset"]["name"],
        "model_params_total": count_parameters(model),
        "model_params_trainable": count_parameters(model, only_trainable=True),
        "num_clients": int(cfg["federated"]["num_clients"]),
        "rounds": int(cfg["federated"]["rounds"]),
        "clients_per_round": int(cfg["federated"]["clients_per_round"]),
        "local_epochs": int(cfg["federated"]["local_epochs"]),
        "max_steps_per_epoch": int(cfg["federated"]["max_steps_per_epoch"]),
        "dirichlet_alpha": float(cfg["federated"]["dirichlet_alpha"]),
        "lr": float(cfg["training"]["lr"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "ema_decay": float(cfg["training"].get("ema_decay", 0.0)),
        "x0_loss_weight": float(cfg["diffusion"].get("x0_loss_weight", 0.0)),
        "semantic_eval_every": int(cfg.get("evaluation", {}).get("semantic_eval_every", 0)),
        "trainable_param_tensors": len(trainable_names),
        "unfreeze_param_prefixes": unfreeze_prefixes,
        "enabled_extra_params": len(enabled_param_names),
        "save_carriers": save_carriers,
    }
    save_json(summary, output_dir / "run_summary.json")

    print("=" * 80)
    print("FedDiffSecure minimal prototype")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)

    # -------------------------
    # 评估配置
    # -------------------------
    eval_cfg = cfg.get("evaluation", {})
    semantic_eval_every = int(eval_cfg.get("semantic_eval_every", 0))
    semantic_eval_enabled = semantic_eval_every > 0 and supports_semantic_metrics(str(cfg["dataset"]["name"]))
    real_eval_images = collect_real_images(
        eval_dataset,
        max_samples=int(eval_cfg.get("num_real_samples", 128)),
        batch_size=int(eval_cfg.get("eval_batch_size", 128)),
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    round_metrics: list[dict] = []
    best_proxy_fid = float("inf")
    best_proxy_round_id = 0
    best_semantic_fid = float("inf")
    best_semantic_round_id = 0
    best_checkpoint_metric = "proxy_fid"
    ema_decay = float(cfg["training"].get("ema_decay", 0.0))
    ema_state: dict[str, torch.Tensor] | None = None
    round_durations_sec: list[float] = []

    # 为了让每轮客户端选择可复现
    selection_rng = random.Random(int(cfg["seed"]))

    num_clients = int(cfg["federated"]["num_clients"])
    clients_per_round = int(cfg["federated"]["clients_per_round"])

    for round_id in range(1, int(cfg["federated"]["rounds"]) + 1):
        round_start_time = time.perf_counter()
        if clients_per_round >= num_clients:
            selected = client_subsets
        else:
            selected = selection_rng.sample(client_subsets, k=clients_per_round)

        client_payloads = []
        print(f"\n[Round {round_id}] selected clients: {[s.client_id for s in selected]}")

        for subset in selected:
            client = FederatedClient(
                client_id=subset.client_id,
                subset=subset,
                cfg=cfg,
                diffusion=diffusion,
                device=device,
            )
            result = client.train(server.get_global_model())

            key = f"{cfg['stego']['key_prefix']}-round{round_id}-client{result.client_id}"
            carrier_path = carriers_dir / f"round_{round_id:03d}_client_{result.client_id:02d}.png"
            carrier_path, package_info = package_state_as_stego_png(
                result.trainable_state,
                output_path=carrier_path,
                key=key,
                seed=int(cfg["seed"]) + round_id * 100 + result.client_id,
            )
            recovered_state, recover_info = recover_state_from_stego_png(carrier_path, key=key)
            verified = package_info["raw_sha256"] == recover_info["raw_sha256"]
            carrier_record_path = str(carrier_path) if save_carriers else ""
            if not save_carriers and carrier_path.exists():
                carrier_path.unlink()

            audit_logger.log(
                {
                    "round": round_id,
                    "client_id": result.client_id,
                    "num_samples": result.num_samples,
                    "avg_loss": result.avg_loss,
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
                    "trainable_state": recovered_state,
                }
            )
            print(
                f"  client={result.client_id} loss={result.avg_loss:.4f} "
                f"samples={result.num_samples} payload={package_info['payload_bytes_raw']}B "
                f"verified={verified}"
            )

        # -------------------------
        # 聚合
        # -------------------------
        aggregation = server.aggregate(round_id=round_id, client_payloads=client_payloads)
        if ema_decay > 0.0:
            if ema_state is None:
                ema_state = _clone_state_dict(server.model)
            else:
                _update_ema_state(ema_state, server.model, decay=ema_decay)

        # -------------------------
        # 每轮评估：proxy_fid
        # -------------------------
        fake_images = server.generate_samples(
            num_images=int(eval_cfg.get("num_fake_samples", 128))
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

        round_record = {
            "round": round_id,
            "avg_client_loss": round(float(aggregation.avg_client_loss), 6),
            "proxy_fid": round(float(proxy_fid), 6),
            "total_samples": int(aggregation.total_samples),
            "aggregated_clients": int(aggregation.num_clients),
        }
        round_duration_sec = time.perf_counter() - round_start_time
        round_record["round_duration_sec"] = round(float(round_duration_sec), 3)
        round_durations_sec.append(round_duration_sec)
        if semantic_metrics is not None:
            round_record.update(semantic_metrics)
        round_metrics.append(round_record)

        if float(proxy_fid) < best_proxy_fid:
            best_proxy_fid = float(proxy_fid)
            best_proxy_round_id = round_id
            torch.save(server.model.state_dict(), output_dir / "global_model_best_proxy.pt")
            if best_semantic_round_id == 0:
                torch.save(server.model.state_dict(), output_dir / "global_model_best.pt")

        if semantic_metrics is not None:
            classifier_fid = float(semantic_metrics["classifier_fid"])
            if classifier_fid < best_semantic_fid:
                best_semantic_fid = classifier_fid
                best_semantic_round_id = round_id
                best_checkpoint_metric = "classifier_fid"
                torch.save(server.model.state_dict(), output_dir / "global_model_best_semantic.pt")
                torch.save(server.model.state_dict(), output_dir / "global_model_best.pt")

        save_json(round_metrics, metrics_dir / "round_metrics.json")
        _write_round_metrics_csv(round_metrics, metrics_dir / "round_metrics.csv")

        summary_line = (
            f"[Round {round_id}] avg_client_loss={aggregation.avg_client_loss:.4f}, "
            f"proxy_fid={proxy_fid:.4f}, total_samples={aggregation.total_samples}, "
            f"aggregated_clients={aggregation.num_clients}, round_sec={round_duration_sec:.1f}"
        )
        if semantic_metrics is not None:
            summary_line += (
                f", classifier_fid={semantic_metrics['classifier_fid']:.2f}, "
                f"conf={semantic_metrics['top1_confidence_mean']:.3f}"
            )
        print(summary_line)

    # -------------------------
    # 保存最终模型
    # -------------------------
    torch.save(server.model.state_dict(), output_dir / "global_model_full.pt")
    torch.save(server.model.state_dict(), output_dir / "global_model_final.pt")
    if ema_state is not None:
        torch.save(ema_state, output_dir / "global_model_ema.pt")

    # 保存最终指标摘要
    if round_metrics:
        best_round = min(round_metrics, key=lambda x: x["proxy_fid"])
        total_wall_clock_sec = time.perf_counter() - run_start_time
        final_metrics = {
            "last_round": round_metrics[-1],
            "best_proxy_fid_round": best_round,
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

    # -------------------------
    # 审计总结
    # -------------------------
    audit_summary = summarize_audit_log(output_dir / "audit_log.jsonl")
    save_json(audit_summary, output_dir / "audit_summary.json")
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    summary["wall_clock_seconds"] = round(float(time.perf_counter() - run_start_time), 3)
    if round_durations_sec:
        summary["avg_round_seconds"] = round(float(sum(round_durations_sec) / len(round_durations_sec)), 3)
        summary["max_round_seconds"] = round(float(max(round_durations_sec)), 3)
    save_json(summary, output_dir / "run_summary.json")

    if bool(cfg["system"].get("save_final_samples", False)):
        server.save_samples(str(samples_dir / "final.png"), num_images=4)
        if (best_semantic_round_id > 0 or best_proxy_round_id > 0) and (output_dir / "global_model_best.pt").exists():
            original_model = server.model
            best_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            best_model.load_state_dict(torch.load(output_dir / "global_model_best.pt", map_location=device))
            server.model = best_model
            server.save_samples(str(samples_dir / "best.png"), num_images=4)
            server.model = original_model
        if ema_state is not None and (output_dir / "global_model_ema.pt").exists():
            original_model = server.model
            ema_model = build_model(cfg["dataset"], cfg["model"]).to(device)
            ema_model.load_state_dict(torch.load(output_dir / "global_model_ema.pt", map_location=device))
            server.model = ema_model
            server.save_samples(str(samples_dir / "ema.png"), num_images=4)
            server.model = original_model

    print("\nDone.")
    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
