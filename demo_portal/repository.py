from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from feddiffsecure.utils import load_yaml

from .markdown_view import markdown_to_html


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CONTENT_ROOT = ROOT / "demo_portal" / "content"
RUNS_REGISTRY_PATH = CONTENT_ROOT / "portal_runs.yaml"
COMPARISON_REGISTRY_PATH = CONTENT_ROOT / "comparison_groups.yaml"
RESEARCH_REGISTRY_PATH = CONTENT_ROOT / "research_library.yaml"
QUEUE_SUMMARY_PATH = OUTPUTS / "launch_logs" / "tiny_sd_lora_long100_suite_summary.json"
QUEUE_CONFIG_LIST_PATH = ROOT / "configs" / "tiny_sd_lora_long100_suite.txt"


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def artifact_relpath(path: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path is outside repository root: {resolved}")
    return str(resolved.relative_to(root_resolved)).replace("\\", "/")


def _artifact_rel_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return artifact_relpath(path)
    except ValueError:
        return ""


def _to_float(value: Any) -> float | None:
    if value in (None, "", "n/a", "nan", "NaN", "Infinity", "-Infinity", "inf", "-inf"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_count(value: Any) -> str:
    number = _to_int(value)
    if number is None:
        return "n/a"
    return f"{number:,}"


def _format_metric(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _format_learning_rate(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    if 0 < abs(number) < 1e-3:
        return f"{number:.1e}"
    return f"{number:g}"


def _format_bytes_zh(value: Any) -> str:
    number = _to_int(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "无跨端传输"
    if number >= 1024 * 1024:
        return f"{number / (1024 * 1024):.2f} MB"
    if number >= 1024:
        return f"{number / 1024:.1f} KB"
    return f"{number} B"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_round_progress(completed: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{completed}/{total} 轮"


def _existing_images(run_dir: Path) -> list[Path]:
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return []

    preferred = [
        "best_loss.png",
        "best_proxy.png",
        "best.png",
        "best_semantic.png",
        "best_compare.png",
        "final.png",
        "ema.png",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()

    for name in preferred:
        candidate = samples_dir / name
        if candidate.exists() and candidate not in seen:
            paths.append(candidate)
            seen.add(candidate)

    for candidate in sorted(samples_dir.glob("round_*.png"), reverse=True):
        if candidate not in seen:
            paths.append(candidate)
            seen.add(candidate)

    return paths


def _summary_from_config(meta: dict[str, Any]) -> dict[str, Any]:
    raw_config_path = meta.get("config_path")
    if not raw_config_path:
        return {}
    config_path = _resolve_path(raw_config_path)
    if not config_path.exists():
        return {}
    payload = load_yaml(config_path)
    federated = payload.get("federated", {})
    lora = payload.get("lora", {})
    training = payload.get("training", {})
    dataset = payload.get("dataset", {})
    audit = payload.get("audit", {})
    model = payload.get("model", {})
    return {
        "config_path": str(config_path),
        "pretrained_model_name_or_path": model.get("pretrained_model_name_or_path"),
        "dataset": dataset.get("name"),
        "resolution": dataset.get("resolution"),
        "training_mode": "federated_lora_sd15" if (_to_int(federated.get("clients_per_round")) or 1) > 1 else "centralized_lora_sd15",
        "dtype": f"torch.{training.get('mixed_precision')}" if training.get("mixed_precision") else None,
        "rounds": federated.get("rounds"),
        "num_clients": federated.get("num_clients"),
        "clients_per_round": federated.get("clients_per_round"),
        "client_epochs": federated.get("client_epochs"),
        "max_steps_per_client": federated.get("max_steps_per_client"),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "target_modules": lora.get("target_modules"),
        "sample_seed": (payload.get("sampling", {}) or {}).get("seed"),
        "lr": training.get("lr"),
        "audit_enabled": audit.get("enabled", False),
    }


def _derive_audit_summary(run_dir: Path, round_rows: list[dict[str, Any]]) -> dict[str, int]:
    summary_path = run_dir / "audit_summary.json"
    existing = _read_json(summary_path)
    if isinstance(existing, dict):
        records = _to_int(existing.get("records")) or 0
        verified = _to_int(existing.get("verified")) or 0
        failed = _to_int(existing.get("failed"))
        if failed is None:
            failed = max(records - verified, 0)
        return {
            "records": records,
            "verified": verified,
            "failed": failed,
        }

    records = 0
    verified = 0
    for row in round_rows:
        records += _to_int(row.get("audit_records")) or 0
        verified += _to_int(row.get("audit_verified")) or 0
    return {
        "records": records,
        "verified": verified,
        "failed": max(records - verified, 0),
    }


def _extract_best_metric(
    final_metrics: dict[str, Any] | None,
    round_rows: list[dict[str, Any]],
) -> tuple[str, float | None, int | None]:
    if final_metrics:
        best_loss = _to_float(final_metrics.get("best_loss"))
        best_loss_round = final_metrics.get("best_loss_round")
        if best_loss is not None:
            return "最优训练损失", best_loss, _to_int((best_loss_round or {}).get("round"))

        best_proxy = final_metrics.get("best_proxy_fid_round")
        if isinstance(best_proxy, dict):
            return "最优 proxy_fid", _to_float(best_proxy.get("proxy_fid")), _to_int(best_proxy.get("round"))

        best_classifier = final_metrics.get("best_classifier_fid_round")
        if isinstance(best_classifier, dict):
            return "最优 classifier_fid", _to_float(best_classifier.get("classifier_fid")), _to_int(best_classifier.get("round"))

    loss_rows = [row for row in round_rows if _to_float(row.get("avg_client_loss")) is not None]
    if loss_rows:
        best_row = min(loss_rows, key=lambda row: float(row["avg_client_loss"]))
        return "最优训练损失", _to_float(best_row.get("avg_client_loss")), _to_int(best_row.get("round"))

    proxy_rows = [row for row in round_rows if _to_float(row.get("proxy_fid")) is not None]
    if proxy_rows:
        best_row = min(proxy_rows, key=lambda row: float(row["proxy_fid"]))
        return "最优 proxy_fid", _to_float(best_row.get("proxy_fid")), _to_int(best_row.get("round"))

    classifier_rows = [row for row in round_rows if _to_float(row.get("classifier_fid")) is not None]
    if classifier_rows:
        best_row = min(classifier_rows, key=lambda row: float(row["classifier_fid"]))
        return "最优 classifier_fid", _to_float(best_row.get("classifier_fid")), _to_int(best_row.get("round"))

    return "待生成", None, None


def _extract_latest_metric(round_rows: list[dict[str, Any]]) -> tuple[str, float | None, int | None]:
    if not round_rows:
        return "当前指标", None, None
    latest = round_rows[-1]
    if _to_float(latest.get("avg_client_loss")) is not None:
        return "当前训练损失", _to_float(latest.get("avg_client_loss")), _to_int(latest.get("round"))
    if _to_float(latest.get("proxy_fid")) is not None:
        return "当前 proxy_fid", _to_float(latest.get("proxy_fid")), _to_int(latest.get("round"))
    if _to_float(latest.get("classifier_fid")) is not None:
        return "当前 classifier_fid", _to_float(latest.get("classifier_fid")), _to_int(latest.get("round"))
    return "当前指标", None, _to_int(latest.get("round"))


def _robust_round_duration(round_rows: list[dict[str, Any]]) -> float | None:
    durations = [_to_float(row.get("round_duration_sec")) for row in round_rows]
    valid = [value for value in durations if value is not None]
    return _mean(valid)


def _derive_status(run_dir: Path, rounds_total: int, completed_rounds: int) -> tuple[str, str]:
    if not run_dir.exists():
        return "pending", "未启动"
    if rounds_total > 0 and completed_rounds >= rounds_total:
        return "completed", "已完成"
    if completed_rounds > 0:
        return "running", "运行中"
    return "initialized", "已创建"


def _mode_from_meta(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    if meta.get("mode"):
        return str(meta["mode"])
    training_mode = str(summary.get("training_mode") or "").lower()
    if "federated" in training_mode:
        return "联邦 LoRA 安全更新"
    return "中心化 LoRA 对照"


def _training_pattern(run: dict[str, Any]) -> str:
    summary = run.get("summary", {})
    num_clients = _to_int(summary.get("num_clients")) or 1
    clients_per_round = _to_int(summary.get("clients_per_round")) or 1
    if clients_per_round <= 1:
        return "单机训练，不发生跨客户端上传或聚合。"
    if clients_per_round >= num_clients and num_clients > 1:
        return f"{num_clients} 个客户端每轮全参与，服务器聚合全部 LoRA 更新。"
    return f"{num_clients} 个客户端中每轮抽取 {clients_per_round} 个参与联邦聚合。"


def _lora_setting(run: dict[str, Any]) -> str:
    summary = run.get("summary", {})
    rank = _format_count(summary.get("lora_rank"))
    alpha = _format_count(summary.get("lora_alpha"))
    modules = summary.get("target_modules") or []
    if isinstance(modules, list) and modules:
        module_text = "、".join(str(item) for item in modules)
        return f"rank={rank}，alpha={alpha}，作用层：{module_text}"
    return f"rank={rank}，alpha={alpha}"


def _budget_setting(run: dict[str, Any]) -> str:
    summary = run.get("summary", {})
    rounds = _format_count(summary.get("rounds"))
    steps = _format_count(summary.get("max_steps_per_client") or summary.get("max_steps_per_epoch"))
    resolution = _format_count(summary.get("resolution"))
    lr = _format_learning_rate(summary.get("lr"))
    return f"{rounds} 轮，单客户端 {steps} 步，分辨率 {resolution}，学习率 {lr}"


def _payload_average(round_rows: list[dict[str, Any]]) -> int | None:
    values = [_to_float(row.get("payload_bytes_compressed_mean")) for row in round_rows]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return int(sum(valid) / len(valid))


def _format_duration_zh(total_seconds: float | None) -> str:
    if total_seconds is None:
        return "n/a"
    seconds = int(max(total_seconds, 0))
    hours, remain = divmod(seconds, 3600)
    minutes, _ = divmod(remain, 60)
    if hours > 0:
        return f"{hours} 小时 {minutes} 分"
    return f"{minutes} 分钟"


def _queue_config_paths() -> list[Path]:
    if not QUEUE_CONFIG_LIST_PATH.exists():
        return []
    items: list[Path] = []
    for raw in QUEUE_CONFIG_LIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(_resolve_path(line))
    return items


def _estimate_total_runtime_from_config(config_path: Path) -> float:
    payload = load_yaml(config_path)
    dataset = payload.get("dataset", {})
    federated = payload.get("federated", {})
    resolution = _to_int(dataset.get("resolution")) or 128
    clients_per_round = _to_int(federated.get("clients_per_round")) or 1
    rounds = _to_int(federated.get("rounds")) or 100

    if resolution >= 256:
        if clients_per_round >= 4:
            per_round = 66.5
        elif clients_per_round >= 2:
            per_round = 18.3
        else:
            per_round = 8.6
    else:
        if clients_per_round >= 2:
            per_round = 10.0
        else:
            per_round = 5.0
        if str(dataset.get("name") or "").lower() == "cifar10":
            per_round += 1.0

    return rounds * per_round


def _queue_eta_bundle() -> dict[str, Any]:
    config_paths = _queue_config_paths()
    summary = _read_json(QUEUE_SUMMARY_PATH)
    completed_jobs: dict[str, dict[str, Any]] = {}
    if isinstance(summary, dict):
        for job in summary.get("jobs", []):
            config_path = str(_resolve_path(job.get("config_path", ""))) if job.get("config_path") else ""
            if config_path:
                completed_jobs[config_path] = dict(job)

    runs = list_runs()
    running_runs = {
        str(_resolve_path(run["summary"].get("config_path"))): run
        for run in runs
        if run["status"] == "running" and run["summary"].get("config_path")
    }

    remaining_seconds = 0.0
    completed_count = 0
    running_count = 0
    pending_count = 0
    active_titles: list[str] = []

    for config_path in config_paths:
        config_key = str(config_path)
        if config_key in running_runs:
            run = running_runs[config_key]
            rounds_total = int(run.get("rounds_total") or 0)
            completed_rounds = int(run.get("completed_rounds") or 0)
            remaining_rounds = max(rounds_total - completed_rounds, 0)
            round_duration = run.get("round_duration_avg")
            if round_duration is None:
                total_estimate = _estimate_total_runtime_from_config(config_path)
                round_duration = total_estimate / rounds_total if rounds_total else 0.0
            remaining_seconds += remaining_rounds * float(round_duration or 0.0)
            running_count += 1
            active_titles.append(run["title"])
            continue

        job = completed_jobs.get(config_key)
        if job and job.get("status") == "completed":
            completed_count += 1
            continue

        pending_count += 1
        remaining_seconds += _estimate_total_runtime_from_config(config_path)

    total_jobs = len(config_paths)
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_count,
        "running_jobs": running_count,
        "pending_jobs": pending_count,
        "active_titles": active_titles,
        "remaining_seconds": remaining_seconds,
        "remaining_text": _format_duration_zh(remaining_seconds),
    }


@lru_cache(maxsize=1)
def _run_registry() -> list[dict[str, Any]]:
    payload = load_yaml(RUNS_REGISTRY_PATH)
    return list(payload.get("runs", []))


@lru_cache(maxsize=1)
def _comparison_registry() -> list[dict[str, Any]]:
    payload = load_yaml(COMPARISON_REGISTRY_PATH)
    return list(payload.get("comparison_groups", []))


@lru_cache(maxsize=1)
def _research_registry() -> dict[str, Any]:
    return load_yaml(RESEARCH_REGISTRY_PATH)


def _run_meta(run_id: str) -> dict[str, Any] | None:
    return next((run for run in _run_registry() if run["id"] == run_id), None)


def _comparison_meta(comparison_id: str) -> dict[str, Any] | None:
    return next((group for group in _comparison_registry() if group["id"] == comparison_id), None)


def get_run(run_id: str) -> dict[str, Any] | None:
    meta = _run_meta(run_id)
    if meta is None:
        return None

    run_dir = _resolve_path(meta["run_dir"])
    config_summary = _summary_from_config(meta)
    summary = {**config_summary, **(_read_json(run_dir / "run_summary.json") or {})}
    final_metrics = _read_json(run_dir / "metrics" / "final_metrics.json") or {}
    round_metrics = _read_json(run_dir / "metrics" / "round_metrics.json")
    if not isinstance(round_metrics, list):
        round_metrics = _read_csv_rows(run_dir / "metrics" / "round_metrics.csv")
    round_metrics = round_metrics or []

    image_paths = [artifact_relpath(path) for path in _existing_images(run_dir)]
    best_metric_name, best_metric_value, best_metric_round = _extract_best_metric(final_metrics, round_metrics)
    current_metric_name, current_metric_value, current_metric_round = _extract_latest_metric(round_metrics)
    audit_summary = _derive_audit_summary(run_dir, round_metrics)
    rounds_total = _to_int(summary.get("rounds")) or _to_int(meta.get("rounds")) or 0
    completed_rounds = _to_int(round_metrics[-1].get("round")) if round_metrics else 0
    completed_rounds = completed_rounds or 0
    progress_ratio = (completed_rounds / rounds_total) if rounds_total else 0.0
    status, status_label = _derive_status(run_dir, rounds_total, completed_rounds)
    payload_avg = _payload_average(round_metrics)
    lora_numel = _to_int(summary.get("lora_numel"))
    round_duration_avg = _robust_round_duration(round_metrics)
    trainable_ratio = None
    if _to_int(summary.get("model_params_total")):
        trainable_ratio = (_to_int(summary.get("model_params_trainable")) or 0) / float(_to_int(summary.get("model_params_total")) or 1)

    if status == "pending":
        metric_story = "这组 100 轮长线实验已经写入门户注册表，等待队列启动后会自动补全指标和样图。"
    elif best_metric_value is not None:
        metric_story = (
            f"当前已完成 {completed_rounds}/{rounds_total or '?'} 轮，"
            f"{best_metric_name}为 {best_metric_value:.4f}"
            f"{f'（第 {best_metric_round} 轮）' if best_metric_round else ''}。"
        )
    else:
        metric_story = f"当前已完成 {completed_rounds}/{rounds_total or '?'} 轮，正在等待首批可展示指标。"

    return {
        "id": run_id,
        "title": meta["title"],
        "dataset": meta["dataset"],
        "mode": _mode_from_meta(meta, summary),
        "role": meta["role"],
        "accent": meta.get("accent", "federated"),
        "summary": summary,
        "final_metrics": final_metrics,
        "recent_rounds": round_metrics[-12:],
        "run_dir": str(run_dir),
        "run_dir_rel": _artifact_rel_if_exists(run_dir),
        "image_paths": image_paths,
        "preview_image_path": image_paths[0] if image_paths else "",
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_metric_round": best_metric_round,
        "best_metric_display": f"{best_metric_name} {best_metric_value:.4f}" if best_metric_value is not None else "待生成",
        "current_metric_name": current_metric_name,
        "current_metric_value": current_metric_value,
        "current_metric_round": current_metric_round,
        "current_metric_display": f"{current_metric_name} {current_metric_value:.4f}" if current_metric_value is not None else "待生成",
        "audit_summary": audit_summary,
        "audit_status_text": (
            f"{audit_summary['verified']}/{audit_summary['records']}"
            if audit_summary["records"]
            else "n/a"
        ),
        "lora_numel": lora_numel,
        "lora_numel_display": _format_count(lora_numel),
        "mean_payload_bytes": payload_avg,
        "payload_display": _format_bytes_zh(payload_avg),
        "round_duration_avg": round_duration_avg,
        "round_duration_display": f"{round_duration_avg:.1f} 秒/轮" if round_duration_avg is not None else "n/a",
        "rounds_total": rounds_total,
        "completed_rounds": completed_rounds,
        "progress_ratio": progress_ratio,
        "progress_text": _format_round_progress(completed_rounds, rounds_total),
        "status": status,
        "status_label": status_label,
        "metric_story": metric_story,
        "trainable_ratio": trainable_ratio,
        "trainable_ratio_display": _format_ratio(trainable_ratio),
        "training_pattern": _training_pattern({"summary": summary}),
        "lora_setting": _lora_setting({"summary": summary}),
        "budget_setting": _budget_setting({"summary": summary}),
        "is_completed": status == "completed",
    }


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for meta in _run_registry():
        run = get_run(meta["id"])
        if run is not None:
            runs.append(run)
    return runs


def _available_runs(run_ids: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_id in run_ids:
        run = get_run(run_id)
        if run is not None:
            runs.append(run)
    return runs


def _highlight_cards(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    available_metric_runs = [run for run in runs if run.get("best_metric_value") is not None]
    available_payload_runs = [run for run in runs if run.get("mean_payload_bytes") is not None]
    available_audit_runs = [run for run in runs if (run.get("audit_summary", {}).get("records") or 0) > 0]

    cards: list[dict[str, str]] = []
    if available_metric_runs:
        best_run = min(available_metric_runs, key=lambda run: float(run["best_metric_value"]))
        cards.append(
            {
                "kicker": "当前最低训练损失",
                "title": best_run["title"],
                "value": best_run["best_metric_display"],
                "note": "用于观察在同一预算下，哪一条训练范式目前优化得更稳。",
            }
        )
    if available_payload_runs:
        payload_run = min(available_payload_runs, key=lambda run: float(run["mean_payload_bytes"]))
        cards.append(
            {
                "kicker": "通信负载最低",
                "title": payload_run["title"],
                "value": f"平均上传 {payload_run['payload_display']}",
                "note": "这项指标直接服务论文里“轻量化更新是否真正减负”的论点。",
            }
        )
    if available_audit_runs:
        audit_run = max(available_audit_runs, key=lambda run: int(run["audit_summary"]["records"]))
        cards.append(
            {
                "kicker": "审计覆盖最多",
                "title": audit_run["title"],
                "value": f"已验证 {audit_run['audit_status_text']}",
                "note": "它最适合展示隐写封装、恢复和完整性校验链路已经真正参与训练过程。",
            }
        )
    return cards


def _comparison_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        rows.append(
            {
                "title": run["title"],
                "mode": run["mode"],
                "status": run["status_label"],
                "best_metric": run["best_metric_display"],
                "progress": run["progress_text"],
                "audit": run["audit_status_text"],
                "payload": run["payload_display"],
            }
        )
    return rows


def _comparison_run_cards(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for run in runs:
        facts = [
            {"label": "训练组织", "value": run["training_pattern"]},
            {"label": "LoRA 设置", "value": run["lora_setting"]},
            {"label": "训练预算", "value": run["budget_setting"]},
            {"label": "当前进度", "value": f"{run['status_label']}，{run['progress_text']}"},
            {"label": "审计状态", "value": run["audit_status_text"]},
            {"label": "平均上传负载", "value": run["payload_display"]},
        ]
        chips = [
            run["status_label"],
            run["mode"],
            run["dataset"],
            f"LoRA {run['lora_numel_display']}",
        ]
        cards.append(
            {
                "run_id": run["id"],
                "title": run["title"],
                "mode": run["mode"],
                "role": run["role"],
                "accent": run["accent"],
                "preview_image_rel": run["preview_image_path"],
                "image_label": "当前样图预览" if run["preview_image_path"] else "等待样图",
                "chips": [chip for chip in chips if chip and chip != "LoRA n/a"],
                "facts": facts,
                "summary_text": run["metric_story"],
            }
        )
    return cards


def get_comparison(comparison_id: str) -> dict[str, Any] | None:
    meta = _comparison_meta(comparison_id)
    if meta is None:
        return None

    runs = _available_runs(list(meta.get("run_ids", [])))
    if not runs:
        return None

    hero_image_rel = next((run["preview_image_path"] for run in runs if run["preview_image_path"]), "")
    reference_title = "当前主展示图"
    reference_description = "对比页优先展示当前已跑出样图的实验。随着 100 轮长线实验继续推进，这里的预览会自动替换为最新样图。"

    return {
        "id": meta["id"],
        "title": meta["title"],
        "dataset": meta.get("dataset", "未分类"),
        "description": meta.get("description", ""),
        "page_note": meta.get("page_note"),
        "highlights": _highlight_cards(runs),
        "metric_columns": [
            {"key": "title", "label": "实验"},
            {"key": "mode", "label": "训练范式"},
            {"key": "status", "label": "状态"},
            {"key": "best_metric", "label": "当前最好指标"},
            {"key": "progress", "label": "进度"},
            {"key": "audit", "label": "审计"},
            {"key": "payload", "label": "平均上传负载"},
        ],
        "metric_rows": _comparison_metric_rows(runs),
        "run_cards": _comparison_run_cards(runs),
        "hero_image_rel": hero_image_rel,
        "reference_image_rel": hero_image_rel,
        "reference_title": reference_title,
        "reference_description": reference_description,
        "file_links": [],
        "report_html": "",
        "report_label": meta.get("report_label", "对比说明"),
    }


def list_comparisons() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for meta in _comparison_registry():
        if not meta.get("visible", True):
            continue
        comparison = get_comparison(meta["id"])
        if comparison is not None:
            groups.append(comparison)
    return groups


def get_reference_file(doc_id: str) -> dict[str, Any] | None:
    registry = _research_registry()
    file_meta = registry.get("reference_files", {}).get(doc_id)
    if not file_meta:
        return None
    file_path = _resolve_path(file_meta["path"])
    if not file_path.exists():
        return None
    return {
        "id": doc_id,
        "label": file_meta.get("label", doc_id),
        "kind": file_meta.get("kind", file_path.suffix.lstrip(".")),
        "path": file_path,
    }


def get_research_bundle() -> dict[str, Any]:
    registry = _research_registry()
    site = registry.get("site", {})
    project_image = _resolve_path(site.get("model_image", ""))
    sections: list[dict[str, Any]] = []

    for section in registry.get("sections", []):
        source_links: list[dict[str, Any]] = []
        for doc_id in section.get("doc_ids", []):
            file_meta = get_reference_file(doc_id)
            if file_meta is None:
                continue
            source_links.append(
                {
                    "label": file_meta["label"],
                    "kind": file_meta["kind"],
                    "doc_id": file_meta["id"],
                }
            )

        sections.append(
            {
                "id": section["id"],
                "title": section["title"],
                "source_label": section.get("source_label", ""),
                "summary": section.get("summary", ""),
                "citation_phrases": list(section.get("citation_phrases", [])),
                "adaptation_notes": list(section.get("adaptation_notes", [])),
                "formulas": list(section.get("formulas", [])),
                "source_links": source_links,
            }
        )

    reference_entries: list[dict[str, Any]] = []
    for doc_id in registry.get("reference_files", {}):
        file_meta = get_reference_file(doc_id)
        if file_meta is None:
            continue
        reference_entries.append(
            {
                "id": file_meta["id"],
                "label": file_meta["label"],
                "kind": file_meta["kind"],
            }
        )

    mainline_runs = _available_runs(_mainline_run_ids())
    experiment_tracks = [
        {
            "title": "主实验矩阵：STL10 四组 100 轮长线",
            "text": "主实验固定在 STL10 上展开，用中心化上界、联邦审计版、全客户端覆盖和 rank=4 消融回答训练效果、覆盖度、轻量化代价三类问题。",
            "items": [run["title"] for run in mainline_runs],
        },
        {
            "title": "泛化实验：Fashion / MNIST / CIFAR-10",
            "text": "这部分不抢主图版面，主要回答 LoRA 联邦安全更新是否能迁移到更简单或不同统计特性的公开数据域。",
            "items": [
                "Fashion-MNIST 联邦 / 中心化长线对照",
                "MNIST 联邦 / 中心化长线对照",
                "CIFAR-10 联邦长线尝试",
            ],
        },
        {
            "title": "安全更新链路：封装、恢复、校验",
            "text": "每轮联邦训练不仅有损失和样图，还记录更新负载、哈希校验和审计通过情况，让系统实现和实验指标落在同一条链路里。",
            "items": [
                "客户端更新隐写封装与恢复",
                "SHA-256 完整性校验",
                "audit_verified / audit_records",
            ],
        },
        {
            "title": "工程实现：注册表驱动的原型门户",
            "text": "门户把实验注册表、对照页、论文资料页和输出目录统一管理，方便把论文写作、实验结果和系统演示连成一套。",
            "items": [
                "portal_runs.yaml",
                "comparison_groups.yaml",
                "research_library.yaml",
            ],
        },
    ]
    metric_cards = [
        {
            "title": "训练效果指标",
            "items": [
                "avg_client_loss：轮次平均训练损失",
                "best_loss：当前最佳训练损失",
                "round_duration_sec：单轮耗时",
            ],
            "note": "主表优先比较 100 轮长线中的损失收敛走势和最佳点位。",
        },
        {
            "title": "聚合效果指标",
            "items": [
                "mean_client_update_l2：客户端更新幅度",
                "aggregation_delta_l2：聚合前后差异",
                "global_lora_l2：全局 LoRA 参数范数",
            ],
            "note": "这一组更适合放在系统实验表格里，解释 LoRA 聚合是否稳定。",
        },
        {
            "title": "通信代价指标",
            "items": [
                "payload_bytes_compressed_mean：平均压缩后上传负载",
                "lora_numel：LoRA 可训练参数规模",
                "clients_per_round：每轮参与客户端数",
            ],
            "note": "这组指标直接服务“轻量化训练”的论文关键词。",
        },
        {
            "title": "审计效果指标",
            "items": [
                "audit_verified / audit_records：审计通过率",
                "原始更新 SHA-256 与恢复更新 SHA-256 是否一致",
                "skipped_steps：异常批次是否被稳定跳过",
            ],
            "note": "当前论文里最稳妥的安全表述，是“验证联邦更新在封装与恢复链路中的完整性”。",
        },
    ]

    return {
        "title": site.get("title", "核心思想与实验设计"),
        "subtitle": site.get("subtitle", ""),
        "citation_note": site.get("citation_note", ""),
        "overview": list(site.get("overview", [])),
        "model_image_rel": _artifact_rel_if_exists(project_image),
        "model_caption": site.get("model_caption", ""),
        "sections": sections,
        "reference_entries": reference_entries,
        "experiment_tracks": experiment_tracks,
        "metric_cards": metric_cards,
    }


def _mainline_run_ids() -> list[str]:
    return [
        "tiny_sd_stl10_lora_central_long100",
        "tiny_sd_stl10_lora_fed_audit_long100",
        "tiny_sd_stl10_lora_fullclients_long100",
        "tiny_sd_stl10_lora_rank4_long100",
    ]


def _story_meta(run_id: str) -> dict[str, Any]:
    mapping: dict[str, dict[str, Any]] = {
        "tiny_sd_stl10_lora_fed_audit_long100": {
            "kicker": "主线联邦实验",
            "positioning": "冻结预训练底座，只训练并上传 UNet attention LoRA 更新，同时把更新封装、恢复和完整性审计真正纳入训练闭环。",
            "strengths": [
                "最贴合论文题目“联邦扩散模型的轻量化训练与安全更新”。",
                "每轮只聚合 LoRA 参数，通信负载远低于全量同步。",
                "能直接展示隐写封装和更新审计的系统价值。",
            ],
            "tradeoffs": [
                "仍然要面对 non-IID 和抽样带来的波动。",
                "生成质量通常不会天然优于中心化上界。",
                "工程链路比单机训练更长，调试复杂度更高。",
            ],
            "presentation_tip": "最适合作为论文主实验先讲，先说明为什么论文选择 LoRA 联邦安全更新主线，再讲这条链路如何落地。",
        },
        "tiny_sd_stl10_lora_central_long100": {
            "kicker": "中心化上界",
            "positioning": "保留相同 LoRA 训练对象和相同开源底座，只移除联邦抽样与跨端传输，用作质量和收敛速度的上界对照。",
            "strengths": [
                "最适合回答“如果没有联邦约束，LoRA 训练本身能做到什么程度”。",
                "优化路径最短，便于解释模型本身的可训练性。",
                "能够把联邦带来的性能差异单独分离出来。",
            ],
            "tradeoffs": [
                "无法展示安全更新、跨端传输和审计链路。",
                "真实多方场景通常不能直接集中原始数据。",
                "只能作为上界和参考，不是论文最终场景。",
            ],
            "presentation_tip": "它最适合放在主实验前面，用来建立“联邦引入后到底损失了多少”的参考系。",
        },
        "tiny_sd_stl10_lora_fullclients_long100": {
            "kicker": "覆盖度消融",
            "positioning": "把每轮参与客户端从 2 个提升到 4 个，观察覆盖度、通信代价和稳定性之间是否存在新的平衡点。",
            "strengths": [
                "适合解释“更多客户端参与是否必然更好”。",
                "审计记录更多，更能展示系统链路的完整性。",
                "便于与主线联邦实验形成清晰消融。",
            ],
            "tradeoffs": [
                "每轮上传更多 LoRA 更新，通信压力更大。",
                "覆盖度更高不等于当前配置下一定更优。",
                "对聚合稳定性和工程效率要求更高。",
            ],
            "presentation_tip": "放在主实验之后讲，专门服务“联邦训练组织方式如何影响结果”的问题。",
        },
        "tiny_sd_stl10_lora_rank4_long100": {
            "kicker": "轻量化消融",
            "positioning": "把 LoRA rank 从 16 压到 4，直接量化“更轻的更新”能省下多少负载，同时会损失多少训练能力。",
            "strengths": [
                "最直接支撑“轻量化训练”的论文关键词。",
                "通信负载下降最明显，适合做表格化比较。",
                "能把 LoRA 作为工程折中的作用讲清楚。",
            ],
            "tradeoffs": [
                "容量过低时，训练效果容易变差。",
                "低 rank 不一定能保住复杂数据集上的表达能力。",
                "需要和主线 rank=16 一起看，结论才完整。",
            ],
            "presentation_tip": "最适合回答“为什么论文要选 LoRA，而不是只讲联邦扩散”这一类问题。",
        },
        "tiny_sd_fashion_lora_fed_long100": {
            "kicker": "跨数据集泛化",
            "positioning": "把主线联邦安全更新迁移到更简单的服饰数据域，观察轻量化联邦更新是否仍然稳定。",
            "strengths": [
                "适合作为跨数据集泛化证据。",
                "训练难度低于自然图像，更容易看出稳定性。",
                "审计链路仍然完整，便于说明方法不是只对 STL10 有效。",
            ],
            "tradeoffs": [
                "图像复杂度有限，不宜单独代表自然图像能力。",
                "更适合支撑“可训练性”而不是最终视觉上限。",
                "仍需和中心化对照一起解释。",
            ],
            "presentation_tip": "适合放在 STL10 主线之后，说明这套方法不是只在单一数据集上成立。",
        },
        "tiny_sd_fashion_lora_central_long100": {
            "kicker": "Fashion 上界对照",
            "positioning": "作为 Fashion-MNIST 的中心化上界，对照联邦版本的训练收敛和样图走势。",
            "strengths": [
                "便于单独比较 Fashion 数据域里的联邦代价。",
                "能辅助解释 LoRA 训练不是因为换数据集才稳定。",
            ],
            "tradeoffs": [
                "没有跨端安全更新链路。",
                "更多承担对照作用，而不是系统亮点。",
            ],
            "presentation_tip": "适合作为 Fashion-MNIST 表格中的参考列。",
        },
        "tiny_sd_mnist_lora_fed_long100": {
            "kicker": "MNIST 泛化",
            "positioning": "用最简单的数据域验证 LoRA 联邦更新和审计链路的基本可训练性。",
            "strengths": [
                "能够快速说明方法在简单场景下也能稳定工作。",
                "适合放在附表或泛化小节中。",
            ],
            "tradeoffs": [
                "视觉上限低，不适合作为论文主图。",
                "更偏“可训练性验证”，不是最终主贡献。",
            ],
            "presentation_tip": "适合作为附加泛化实验，不必抢主线版面。",
        },
        "tiny_sd_mnist_lora_central_long100": {
            "kicker": "MNIST 上界对照",
            "positioning": "作为最简单数据域上的中心化对照，帮助区分联邦因素和任务难度因素。",
            "strengths": [
                "对照关系非常清晰。",
                "便于在消融表里解释联邦引入后的额外约束。",
            ],
            "tradeoffs": [
                "工程意义弱于联邦版本。",
                "主要承担补充说明作用。",
            ],
            "presentation_tip": "更适合放在补充表格里，不必占主页面过多篇幅。",
        },
        "tiny_sd_cifar10_lora_fed_long100": {
            "kicker": "开放尝试",
            "positioning": "把同一套联邦 LoRA 安全更新链路扩展到 CIFAR-10，作为额外的开源图像数据集尝试。",
            "strengths": [
                "能证明这套方案不只局限于 STL10。",
                "更贴近自然图像分类数据域。",
            ],
            "tradeoffs": [
                "如果队列尚未跑完，当前更适合写成“持续进行中的尝试”。",
                "结果解释应放在泛化章节，不建议替代主线对照。",
            ],
            "presentation_tip": "最适合写成“额外数据集扩展实验”，作为方法泛化性的补充证据。",
        },
    }
    return mapping.get(
        run_id,
        {
            "kicker": "长线实验",
            "positioning": "这组实验用于补充 LoRA 联邦安全更新主线下的对照和泛化观察。",
            "strengths": ["已纳入统一的 100 轮长线实验框架。"],
            "tradeoffs": ["当前更适合作为补充说明。"],
            "presentation_tip": "适合放在补充分析中介绍。",
        },
    )


def _stl10_comparison_rows() -> list[dict[str, Any]]:
    central = get_run("tiny_sd_stl10_lora_central_long100")
    fed = get_run("tiny_sd_stl10_lora_fed_audit_long100")
    full = get_run("tiny_sd_stl10_lora_fullclients_long100")
    rank4 = get_run("tiny_sd_stl10_lora_rank4_long100")
    labels = [
        ("中心化对照", central),
        ("联邦审计版", fed),
        ("全客户端版", full),
        ("rank=4 消融", rank4),
    ]
    return [
        {
            "dimension": "训练组织",
            "cells": [{"label": label, "value": run["training_pattern"] if run else "待启动"} for label, run in labels],
        },
        {
            "dimension": "LoRA 配置",
            "cells": [{"label": label, "value": run["lora_setting"] if run else "待启动"} for label, run in labels],
        },
        {
            "dimension": "当前最好指标",
            "cells": [{"label": label, "value": run["best_metric_display"] if run else "待生成"} for label, run in labels],
        },
        {
            "dimension": "审计状态",
            "cells": [{"label": label, "value": run["audit_status_text"] if run else "n/a"} for label, run in labels],
        },
        {
            "dimension": "平均上传负载",
            "cells": [{"label": label, "value": run["payload_display"] if run else "n/a"} for label, run in labels],
        },
        {
            "dimension": "论文里回答的问题",
            "cells": [
                {"label": "中心化对照", "value": "LoRA 训练本身在没有联邦约束时能到什么水平。"},
                {"label": "联邦审计版", "value": "轻量化联邦安全更新是否真的可训练、可传输、可审计。"},
                {"label": "全客户端版", "value": "更高覆盖度是否值得更高的通信和聚合开销。"},
                {"label": "rank=4 消融", "value": "更轻的 LoRA 更新是否会明显牺牲训练效果。"},
            ],
        },
    ]


def get_fashion_bundle() -> dict[str, Any]:
    run_ids = [
        "tiny_sd_stl10_lora_fed_audit_long100",
        "tiny_sd_stl10_lora_central_long100",
        "tiny_sd_stl10_lora_fullclients_long100",
        "tiny_sd_stl10_lora_rank4_long100",
        "tiny_sd_fashion_lora_fed_long100",
        "tiny_sd_fashion_lora_central_long100",
        "tiny_sd_mnist_lora_fed_long100",
        "tiny_sd_mnist_lora_central_long100",
        "tiny_sd_cifar10_lora_fed_long100",
    ]
    runs = _available_runs(run_ids)
    active_run = next((run for run in runs if run["status"] == "running"), runs[0] if runs else None)
    comparison = get_comparison("lora_stl10_long100_mainline") or {}
    total_records = sum(int(run["audit_summary"]["records"]) for run in runs)
    total_verified = sum(int(run["audit_summary"]["verified"]) for run in runs)

    cards: list[dict[str, Any]] = []
    for run in runs:
        story = _story_meta(run["id"])
        cards.append(
            {
                "run": run,
                "kicker": story["kicker"],
                "positioning": story["positioning"],
                "training_pattern": run["training_pattern"],
                "observation": run["metric_story"],
                "audit_note": (
                    "这组实验当前没有跨客户端审计记录。"
                    if run["audit_summary"]["records"] == 0
                    else f"当前累计审计通过 {run['audit_status_text']}。"
                ),
                "presentation_tip": story["presentation_tip"],
                "strengths": story["strengths"],
                "tradeoffs": story["tradeoffs"],
            }
        )

    summary_stats = [
        {
            "label": "已注册长线实验",
            "value": str(len(runs)),
            "detail": "门户当前统一围绕 100 轮长线实验组织，不再混入短线演示结果。",
        },
        {
            "label": "覆盖数据集",
            "value": str(len({run['dataset'] for run in runs})),
            "detail": "当前围绕 STL10、Fashion-MNIST、MNIST 和 CIFAR-10 组织主线与泛化实验。",
        },
        {
            "label": "当前运行中",
            "value": str(sum(1 for run in runs if run["status"] == "running")),
            "detail": "队列会按注册表顺序继续推进，其余实验会自动接入门户。",
        },
        {
            "label": "累计审计通过",
            "value": f"{total_verified}/{total_records}" if total_records else "n/a",
            "detail": "这里统计的是当前已跑出结果的联邦长线实验所记录的更新审计条目。",
        },
    ]

    return {
        "hero_title": "100 轮长线实验专题：LoRA 联邦安全更新",
        "focus_question": "围绕“冻结底座、只聚合 UNet-LoRA 更新”这条主线，系统比较中心化上界、联邦审计版、覆盖度消融、rank 消融与跨数据集泛化。",
        "one_liner": "这页把论文最重要的 100 轮长线实验放到同一页里统一讲述，方便把主实验、消融和泛化放在同一条叙事里。",
        "summary_stats": summary_stats,
        "focus_run": active_run,
        "narrative_steps": [
            "先说明论文主线聚焦“联邦场景下的 LoRA 轻量化训练与安全更新”。",
            "再用 STL10 中心化对照、联邦审计版、全客户端版和 rank=4 消融讲清楚主实验和两个关键消融。",
            "随后再看 Fashion-MNIST、MNIST 和 CIFAR-10，说明这条训练链路不是只对一个数据集成立。",
            "最后把审计通过率、平均上传负载和样图一起放回到系统实现层面，形成完整论文叙事。",
        ],
        "cards": cards,
        "comparison": comparison,
        "comparison_rows": _stl10_comparison_rows(),
        "takeaways": [
            {
                "title": "论文主线先讲 STL10",
                "text": "STL10 更适合承担主图、主表和主对照任务，因为它同时承载了视觉质量、联邦训练和安全更新三层信息。",
            },
            {
                "title": "Fashion / MNIST 更适合作为泛化证据",
                "text": "这两类数据集更适合支撑“LoRA 联邦安全更新在简单数据域上也可稳定工作”的论点，而不是直接代替主图。",
            },
            {
                "title": "CIFAR-10 是额外开放尝试",
                "text": "它适合写成额外开源图像数据集扩展，证明方法具备继续外推到更多公开数据集的潜力。",
            },
        ],
    }


def get_live_monitor() -> dict[str, Any]:
    runs = list_runs()
    queue_eta = _queue_eta_bundle()
    running = [run for run in runs if run["status"] == "running"]
    target = running[0] if running else (runs[0] if runs else None)
    if target is None:
        return {
            "title": "暂无活动实验",
            "status_label": "未启动",
            "latest_round": 0,
            "rounds_total": 0,
            "progress_ratio": 0.0,
            "best_metric_display": "待生成",
            "audit_status_text": "n/a",
            "payload_display": "n/a",
            "round_duration_display": "n/a",
            "current_metric_display": "待生成",
            "progress_text": "0 / 0",
            "preview_image_path": "",
            "recent_rounds": [],
            "run_id": "",
            "dataset": "",
            "mode": "",
            "queue_remaining_text": queue_eta["remaining_text"],
            "queue_completed_jobs": queue_eta["completed_jobs"],
            "queue_total_jobs": queue_eta["total_jobs"],
            "message": f"当前没有活动实验。整条长线队列预计还需 {queue_eta['remaining_text']}。",
        }

    message = (
        f"{target['title']} 当前处于 {target['status_label']} 状态，"
        f"平均每轮约 {target['round_duration_display']}，"
        f"{target['best_metric_display']}。"
    )
    if queue_eta["total_jobs"]:
        message += (
            f" 长线队列已完成 {queue_eta['completed_jobs']}/{queue_eta['total_jobs']} 组，"
            f"预计还需 {queue_eta['remaining_text']}。"
        )
    return {
        "title": target["title"],
        "status_label": target["status_label"],
        "latest_round": target["completed_rounds"],
        "rounds_total": target["rounds_total"],
        "progress_ratio": target["progress_ratio"],
        "best_metric_display": target["best_metric_display"],
        "current_metric_display": target["current_metric_display"],
        "audit_status_text": target["audit_status_text"],
        "payload_display": target["payload_display"],
        "round_duration_display": target["round_duration_display"],
        "progress_text": target["progress_text"],
        "preview_image_path": target["preview_image_path"],
        "recent_rounds": target["recent_rounds"],
        "run_id": target["id"],
        "dataset": target["dataset"],
        "mode": target["mode"],
        "queue_remaining_text": queue_eta["remaining_text"],
        "queue_completed_jobs": queue_eta["completed_jobs"],
        "queue_total_jobs": queue_eta["total_jobs"],
        "message": message,
    }


def get_audit_bundle() -> dict[str, Any]:
    run = get_run("tiny_sd_stl10_lora_fed_audit_long100") or {}
    audit_summary = run.get("audit_summary", {"records": 0, "verified": 0, "failed": 0})
    notes_md = f"""
- 当前主线审计实验：**{run.get('title', 'STL10 联邦安全更新长线')}**  
- 已累计记录 **{audit_summary.get('records', 0)}** 条客户端更新，验证通过 **{audit_summary.get('verified', 0)}** 条。  
- 当前门户展示的审计口径是：验证客户端上传的 LoRA 更新在隐写封装、恢复和聚合前后是否保持端到端完整。  
- 这套原型当前证明的是 **传输完整性**，而不是对强对手模型下机密性、抗攻击性的完整形式化证明。  
- 因此论文里最稳妥的写法应当是：**系统验证了联邦更新在安全封装与恢复链路中的完整性**。
"""
    return {
        "audit_summary": audit_summary,
        "notes_html": markdown_to_html(notes_md),
    }


def get_paper_bundle() -> dict[str, Any]:
    pages = list_comparisons()
    mainline_runs = _available_runs(_mainline_run_ids())
    registry_md = "\n".join(
        [
            "- `STL10 100轮主对照`：中心化上界、联邦审计版、全客户端覆盖和 rank=4 消融。",
            "- `跨数据集泛化`：在 STL10、Fashion-MNIST、MNIST、CIFAR-10 上比较 LoRA 联邦安全更新的可训练性。",
            "- `Fashion-MNIST 100轮对照`：保留一组更轻数据域的联邦/中心化长线比较。",
        ]
    )
    storyline_md = """
### 论文主线建议

1. 先明确论文聚焦的是“冻结底座，只聚合 UNet-LoRA 更新”的联邦轻量化训练与安全更新原型。  
2. 再说明为什么 LoRA 更新比全量同步更适合联邦场景下的扩散模型落地。  
3. 接着用 STL10 100 轮长线主对照说明：联邦训练、覆盖度变化和 LoRA 容量变化分别会带来什么影响。  
4. 最后再用 Fashion-MNIST、MNIST 与 CIFAR-10 说明这条链路具备跨数据集泛化能力。  

### 最适合写进论文表格的指标

- 训练效果：`avg_client_loss`、`best_loss`
- 聚合效果：`mean_client_update_l2`、`aggregation_delta_l2`、`global_lora_l2`
- 通信代价：`payload_bytes_compressed_mean`
- 审计效果：`audit_verified / audit_records`
"""
    deliverables = [
        {
            "title": "主图建议",
            "text": "优先使用 STL10 100 轮主对照页里的主图和四组样图卡片，承担论文里的主视觉展示任务。",
        },
        {
            "title": "主表建议",
            "text": "主表固定比较 STL10 中心化上界、联邦审计版、全客户端版和 rank=4 消融，指标围绕训练效果、通信负载与审计结果展开。",
        },
        {
            "title": "补充实验建议",
            "text": "Fashion-MNIST、MNIST 与 CIFAR-10 更适合作为补充表格或泛化小节，支撑方法的可迁移性而不是替代主图。",
        },
    ]
    recommended_runs = []
    for run in mainline_runs:
        recommended_runs.append(
            {
                "title": run["title"],
                "kicker": _story_meta(run["id"])["kicker"],
                "best_metric": run["best_metric_display"],
                "payload": run["payload_display"],
                "audit": run["audit_status_text"],
                "role": run["role"],
            }
        )
    return {
        "pages": pages,
        "registry_html": markdown_to_html(registry_md),
        "storyline_html": markdown_to_html(storyline_md),
        "reference_entries": get_research_bundle()["reference_entries"],
        "deliverables": deliverables,
        "recommended_runs": recommended_runs,
    }


def _markdown_table(columns: list[dict[str, str]], rows: list[dict[str, Any]]) -> str:
    headers = "| " + " | ".join(column["label"] for column in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = [
        "| " + " | ".join(str(row.get(column["key"], "—")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([headers, divider, *body_lines])


def get_paper_table_bundle() -> dict[str, Any]:
    runs = list_runs()
    live_monitor = get_live_monitor()
    mainline_runs = _available_runs(_mainline_run_ids())

    mainline_columns = [
        {"key": "experiment", "label": "实验"},
        {"key": "pattern", "label": "训练组织"},
        {"key": "lora", "label": "LoRA 设置"},
        {"key": "best_metric", "label": "当前最好指标"},
        {"key": "payload", "label": "平均上传负载"},
        {"key": "audit", "label": "审计状态"},
        {"key": "status", "label": "状态"},
    ]
    mainline_rows = [
        {
            "experiment": run["title"],
            "pattern": run["training_pattern"],
            "lora": run["lora_setting"],
            "best_metric": run["best_metric_display"],
            "payload": run["payload_display"],
            "audit": run["audit_status_text"],
            "status": run["status_label"],
        }
        for run in mainline_runs
    ]

    dataset_specs = [
        {
            "dataset": "STL10",
            "central": "tiny_sd_stl10_lora_central_long100",
            "federated": "tiny_sd_stl10_lora_fed_audit_long100",
            "note": "主实验数据集，承担主图与主表。",
        },
        {
            "dataset": "Fashion-MNIST",
            "central": "tiny_sd_fashion_lora_central_long100",
            "federated": "tiny_sd_fashion_lora_fed_long100",
            "note": "更适合做简单数据域下的泛化对照。",
        },
        {
            "dataset": "MNIST",
            "central": "tiny_sd_mnist_lora_central_long100",
            "federated": "tiny_sd_mnist_lora_fed_long100",
            "note": "更偏可训练性验证，不抢主图位置。",
        },
        {
            "dataset": "CIFAR-10",
            "central": "",
            "federated": "tiny_sd_cifar10_lora_fed_long100",
            "note": "额外开放尝试，用于扩展说明。",
        },
    ]
    dataset_columns = [
        {"key": "dataset", "label": "数据集"},
        {"key": "central_best", "label": "中心化最好指标"},
        {"key": "federated_best", "label": "联邦最好指标"},
        {"key": "payload", "label": "联邦平均上传负载"},
        {"key": "audit", "label": "联邦审计状态"},
        {"key": "note", "label": "论文定位"},
    ]
    dataset_rows: list[dict[str, str]] = []
    for spec in dataset_specs:
        central_run = get_run(spec["central"]) if spec["central"] else None
        federated_run = get_run(spec["federated"]) if spec["federated"] else None
        dataset_rows.append(
            {
                "dataset": spec["dataset"],
                "central_best": central_run["best_metric_display"] if central_run else "—",
                "federated_best": federated_run["best_metric_display"] if federated_run else "—",
                "payload": federated_run["payload_display"] if federated_run else "—",
                "audit": federated_run["audit_status_text"] if federated_run else "—",
                "note": spec["note"],
            }
        )

    summary_cards = [
        {
            "label": "已完成长线实验",
            "value": str(sum(1 for run in runs if run["status"] == "completed")),
            "detail": f"当前共登记 {len(runs)} 组长线实验。",
        },
        {
            "label": "当前活动实验",
            "value": live_monitor["title"],
            "detail": f"{live_monitor['latest_round']}/{live_monitor['rounds_total']}，预计还需 {live_monitor['queue_remaining_text']}。",
        },
        {
            "label": "当前累计审计",
            "value": f"{sum(run['audit_summary']['verified'] for run in runs)}/{sum(run['audit_summary']['records'] for run in runs)}",
            "detail": "这里累计的是当前所有已跑出联邦结果的更新审计记录。",
        },
    ]
    notes = [
        "主表建议优先比较 STL10 中心化、联邦审计版、全客户端版和 rank=4 消融四组实验。",
        "跨数据集表建议把 STL10、Fashion-MNIST、MNIST 与 CIFAR-10 放在同一张表里，突出泛化与可迁移性。",
        "当前门户里的“最好指标”统一使用训练损失口径，便于在 LoRA 联邦长线实验之间横向比较。",
    ]

    return {
        "title": "正式论文表格页",
        "subtitle": "自动汇总当前已经跑出的长线结果，生成可直接放入论文正文或附录的正式表格。",
        "summary_cards": summary_cards,
        "mainline_columns": mainline_columns,
        "mainline_rows": mainline_rows,
        "dataset_columns": dataset_columns,
        "dataset_rows": dataset_rows,
        "mainline_markdown": _markdown_table(mainline_columns, mainline_rows),
        "dataset_markdown": _markdown_table(dataset_columns, dataset_rows),
        "notes": notes,
    }


def get_overview() -> dict[str, Any]:
    runs = list_runs()
    comparisons = list_comparisons()
    live_monitor = get_live_monitor()
    queue_eta = _queue_eta_bundle()
    verified = sum(run["audit_summary"]["verified"] for run in runs)
    records = sum(run["audit_summary"]["records"] for run in runs)

    summary_stats = [
        {
            "label": "长线实验",
            "value": str(len(runs)),
            "detail": "门户已经统一切换为 100 轮长线主线，重点展示主实验、消融和泛化结果。",
        },
        {
            "label": "数据集覆盖",
            "value": str(len({run['dataset'] for run in runs})),
            "detail": "当前围绕 STL10、Fashion-MNIST、MNIST、CIFAR-10 组织主线与泛化实验。",
        },
        {
            "label": "专题对比页",
            "value": str(len(comparisons)),
            "detail": "对比页优先服务论文里的主对照、跨数据集泛化和轻量化消融。",
        },
        {
            "label": "预计剩余时间",
            "value": queue_eta["remaining_text"],
            "detail": (
                f"当前长线队列已完成 {queue_eta['completed_jobs']}/{queue_eta['total_jobs']} 组；"
                f"累计审计通过 {verified}/{records}" if queue_eta["total_jobs"] else f"累计审计通过 {verified}/{records}"
            ),
        },
    ]

    guide_cards = [
        {
            "title": "先讲主线转向",
            "text": "论文现在更适合直接表述为“联邦场景下的 LoRA 轻量化训练与安全更新”，把问题边界放在可落地的训练对象和安全更新链路上。",
        },
        {
            "title": "主实验只看 STL10 100 轮",
            "text": "中心化上界、联邦审计版、全客户端版和 rank=4 消融，构成最关键的主实验矩阵。",
        },
        {
            "title": "再补跨数据集泛化",
            "text": "Fashion-MNIST、MNIST 和 CIFAR-10 更适合写成泛化与可迁移性章节，而不是主图替代品。",
        },
    ]

    return {
        "summary_stats": summary_stats,
        "mainline_points": [
            "冻结预训练扩散底座，只在 UNet attention 层训练 LoRA 适配器。",
            "客户端只上传 LoRA 更新，服务器只对对应的低秩矩阵做按样本量加权平均。",
            "将更新封装、恢复和哈希校验纳入同一条联邦训练链路，验证安全更新的端到端完整性。",
        ],
        "guide_cards": guide_cards,
        "live_monitor": live_monitor,
    }


def get_federation_bundle() -> dict[str, Any]:
    focus_runs = _available_runs(_mainline_run_ids())
    focus_cards: list[dict[str, Any]] = []
    for run in focus_runs:
        story = _story_meta(run["id"])
        focus_cards.append(
            {
                "run": run,
                "kicker": story["kicker"],
                "positioning": story["positioning"],
                "points": [
                    run["training_pattern"],
                    run["lora_setting"],
                    run["metric_story"],
                ],
            }
        )

    return {
        "title": "LoRA 联邦安全更新机制：论文里到底要讲什么",
        "subtitle": "这页专门解释冻结底座、训练 UNet-LoRA、联邦聚合安全更新这一条更可落地的论文主线。",
        "one_liner": "联邦学习在这篇论文里的核心作用，不是把样图做得比中心化更好，而是让扩散模型的轻量化更新、跨端传输、聚合与审计成为一个完整系统问题。",
        "focus_cards": focus_cards,
        "pipeline_points": [
            "客户端保留本地数据，只训练 UNet attention 层中的 LoRA 参数。",
            "VAE 和文本编码器保持冻结，避免把联邦训练问题重新推回全量大模型同步。",
            "客户端上传的是 LoRA 低秩更新，而不是整个 UNet 主干权重。",
            "服务器端按样本量加权平均对应层的 LoRA 矩阵，并记录每轮更新的负载与审计结果。",
            "安全部分当前验证的是更新在封装与恢复链路中的完整性，而不是更强的机密性形式化证明。",
        ],
        "formula_cards": [
            {
                "title": "LoRA 参数化",
                "expression": r'W = W_0 + \Delta W,\ \Delta W = \frac{\alpha}{r}BA',
                "explanation": "原始预训练权重保持冻结，只训练低秩适配器矩阵 A、B，从而把联邦上传对象缩小到可控范围。",
            },
            {
                "title": "联邦加权聚合",
                "expression": r'\bar{A} = \sum_k \frac{n_k}{N} A^{(k)},\ \bar{B} = \sum_k \frac{n_k}{N} B^{(k)}',
                "explanation": "服务器不再聚合整网参数，而是只对各客户端上传的 LoRA 矩阵做按数据量加权平均。",
            },
            {
                "title": "潜空间扩散训练目标",
                "expression": r'L = \mathbb{E}\|\epsilon_\theta(z_t, t, c) - \epsilon\|_2^2',
                "explanation": "生成训练仍然遵循扩散模型的噪声预测目标，但联邦优化对象被收缩到 UNet-LoRA。",
            },
            {
                "title": "更新完整性校验",
                "expression": r'\mathrm{SHA256}(u_{\mathrm{pack}}) = \mathrm{SHA256}(u_{\mathrm{recover}})',
                "explanation": "审计模块当前证明的是：上传的 LoRA 更新在隐写封装和恢复后保持端到端一致。",
            },
        ],
        "comparison_rows": _stl10_comparison_rows(),
        "takeaways": [
            {
                "title": "为什么不再主打全量训练",
                "text": "扩散模型的通信代价、优化器状态和激活内存都太重，LoRA 才是联邦落地更合理的训练对象。",
            },
            {
                "title": "为什么要保留中心化对照",
                "text": "中心化上界能把“联邦额外约束带来的损失”单独分离出来，是主实验必不可少的参考系。",
            },
            {
                "title": "为什么要做审计",
                "text": "论文主线不只关心能不能训，还关心客户端更新在安全封装和恢复后能否被可靠验证。",
            },
        ],
    }
