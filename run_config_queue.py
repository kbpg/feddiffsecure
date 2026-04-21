from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from feddiffsecure.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially run multiple training configs with logging.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config path to enqueue. Can be provided multiple times.",
    )
    parser.add_argument(
        "--config-list",
        type=str,
        default=None,
        help="Text file containing one config path per line. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Queue name used for launch logs and summary files. Defaults to timestamp.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch child runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs even if one config fails.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Pass --restart to each child run.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Do not auto-skip configs whose output dir already has final_metrics.json.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["auto", "centralized", "federated", "sd_lora"],
        default="auto",
        help="Launcher selection. 'auto' infers the training script from the config structure.",
    )
    return parser.parse_args()


def _load_config_list(path: Path) -> list[str]:
    configs: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(line)
    return configs


def _collect_configs(args: argparse.Namespace) -> list[Path]:
    configs: list[str] = list(args.config)
    if args.config_list:
        configs.extend(_load_config_list(Path(args.config_list)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in configs:
        path = Path(item).resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise SystemExit("No configs were provided. Use --config or --config-list.")
    return unique


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_entry(config_path: Path, cfg: dict, queue_name: str, index: int, logs_dir: Path) -> dict:
    output_dir = Path(cfg["system"]["output_dir"]).resolve()
    metrics_dir = output_dir / "metrics"
    final_metrics_path = metrics_dir / "final_metrics.json"
    round_metrics_path = metrics_dir / "round_metrics.json"
    stem = config_path.stem
    log_prefix = f"{queue_name}_{index:02d}_{stem}"
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "final_metrics_path": str(final_metrics_path),
        "round_metrics_path": str(round_metrics_path),
        "stdout_log": str(logs_dir / f"{log_prefix}.out.log"),
        "stderr_log": str(logs_dir / f"{log_prefix}.err.log"),
    }


def _resolve_launcher(cfg: dict, launcher_override: str) -> str:
    if launcher_override == "centralized":
        return "run_centralized.py"
    if launcher_override == "federated":
        return "run_federated.py"
    if launcher_override == "sd_lora":
        return "run_sd15_lora_federated.py"
    if isinstance(cfg.get("model"), dict) and "pretrained_model_name_or_path" in cfg.get("model", {}):
        return "run_sd15_lora_federated.py"
    if "centralized" in cfg:
        return "run_centralized.py"
    if "federated" in cfg:
        return "run_federated.py"
    raise ValueError("Unable to infer launcher from config. Expected either a 'centralized' or 'federated' section.")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    logs_dir = ensure_dir(repo_root / "outputs" / "launch_logs")
    queue_name = args.name or f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    queue_summary_path = logs_dir / f"{queue_name}_summary.json"
    configs = _collect_configs(args)

    queue_summary = {
        "queue_name": queue_name,
        "started_at": datetime.now().astimezone().isoformat(),
        "host_python": args.python,
        "restart": bool(args.restart),
        "skip_completed": not args.no_skip_completed,
        "continue_on_error": bool(args.continue_on_error),
        "jobs": [],
    }
    save_json(queue_summary, queue_summary_path)

    completed_jobs = 0
    failed_jobs = 0
    skipped_jobs = 0

    for index, config_path in enumerate(configs, start=1):
        if not config_path.exists():
            job = {
                "config_path": str(config_path),
                "status": "missing_config",
                "completed_at": datetime.now().astimezone().isoformat(),
            }
            queue_summary["jobs"].append(job)
            failed_jobs += 1
            save_json(queue_summary, queue_summary_path)
            if not args.continue_on_error:
                break
            continue

        cfg = load_yaml(config_path)
        job = _build_entry(config_path, cfg, queue_name, index, logs_dir)
        job["launcher"] = _resolve_launcher(cfg, args.launcher)
        job["started_at"] = datetime.now().astimezone().isoformat()
        final_metrics_path = Path(job["final_metrics_path"])

        if final_metrics_path.exists() and not args.restart and not args.no_skip_completed:
            final_metrics = _read_json(final_metrics_path)
            job["status"] = "skipped_completed"
            job["completed_at"] = datetime.now().astimezone().isoformat()
            if isinstance(final_metrics, dict):
                best_semantic = final_metrics.get("best_classifier_fid_round")
                if isinstance(best_semantic, dict):
                    job["best_classifier_fid"] = best_semantic.get("classifier_fid")
                    job["best_classifier_round"] = best_semantic.get("round")
                best_proxy = final_metrics.get("best_proxy_fid_round")
                if isinstance(best_proxy, dict):
                    job["best_proxy_fid"] = best_proxy.get("proxy_fid")
                    job["best_proxy_round"] = best_proxy.get("round")
            queue_summary["jobs"].append(job)
            skipped_jobs += 1
            save_json(queue_summary, queue_summary_path)
            print(f"[queue] skip completed: {config_path}")
            continue

        out_log = Path(job["stdout_log"])
        err_log = Path(job["stderr_log"])
        command = [args.python, job["launcher"], "--config", str(config_path)]
        if args.restart and job["launcher"] == "run_centralized.py":
            command.append("--restart")

        print(f"[queue] start {index}/{len(configs)}: {config_path}")
        launch_started = time.perf_counter()
        with open(out_log, "w", encoding="utf-8") as stdout_f, open(err_log, "w", encoding="utf-8") as stderr_f:
            process = subprocess.run(command, cwd=repo_root, stdout=stdout_f, stderr=stderr_f)
        duration_sec = time.perf_counter() - launch_started

        job["exit_code"] = int(process.returncode)
        job["duration_sec"] = round(float(duration_sec), 3)
        job["completed_at"] = datetime.now().astimezone().isoformat()

        final_metrics = _read_json(final_metrics_path)
        if process.returncode == 0 and isinstance(final_metrics, dict):
            job["status"] = "completed"
            best_semantic = final_metrics.get("best_classifier_fid_round")
            if isinstance(best_semantic, dict):
                job["best_classifier_fid"] = best_semantic.get("classifier_fid")
                job["best_classifier_round"] = best_semantic.get("round")
            best_proxy = final_metrics.get("best_proxy_fid_round")
            if isinstance(best_proxy, dict):
                job["best_proxy_fid"] = best_proxy.get("proxy_fid")
                job["best_proxy_round"] = best_proxy.get("round")
            last_round = final_metrics.get("last_round")
            if isinstance(last_round, dict):
                job["last_round"] = last_round.get("round")
            completed_jobs += 1
        else:
            job["status"] = "failed"
            failed_jobs += 1

        queue_summary["jobs"].append(job)
        save_json(queue_summary, queue_summary_path)
        print(f"[queue] {job['status']}: {config_path}")

        if job["status"] == "failed" and not args.continue_on_error:
            break

    queue_summary["completed_at"] = datetime.now().astimezone().isoformat()
    queue_summary["completed_jobs"] = int(completed_jobs)
    queue_summary["failed_jobs"] = int(failed_jobs)
    queue_summary["skipped_jobs"] = int(skipped_jobs)
    queue_summary["total_jobs"] = len(queue_summary["jobs"])
    save_json(queue_summary, queue_summary_path)
    print(f"[queue] summary saved to {queue_summary_path}")


if __name__ == "__main__":
    main()
