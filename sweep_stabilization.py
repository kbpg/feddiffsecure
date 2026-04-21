from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from feddiffsecure.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small stabilization sweep.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/stabilization.yaml",
        help="Base YAML config to clone for each sweep run.",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[3e-4, 5e-4, 7e-4],
        help="Learning rates to sweep.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        nargs="+",
        default=[20, 30],
        help="max_steps_per_epoch values to sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./outputs/stabilization_sweep",
        help="Root directory for generated configs, logs, and summaries.",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=10,
        help="Number of trailing rounds used for stability statistics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even if final metrics already exist.",
    )
    return parser.parse_args()


def slugify_lr(value: float) -> str:
    return f"{value:.4g}".replace(".", "p")


def build_run_name(lr: float, max_steps: int) -> str:
    return f"lr_{slugify_lr(lr)}_steps_{max_steps}"


def write_yaml(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def summarize_run(run_dir: Path, lr: float, max_steps: int, tail_window: int) -> dict:
    round_metrics_path = run_dir / "metrics" / "round_metrics.json"
    final_metrics_path = run_dir / "metrics" / "final_metrics.json"

    round_metrics = json.loads(round_metrics_path.read_text(encoding="utf-8"))
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))

    proxy_fids = [record["proxy_fid"] for record in round_metrics]
    tail = proxy_fids[-min(tail_window, len(proxy_fids)) :]
    best_record = final_metrics["best_proxy_fid_round"]
    last_record = final_metrics["last_round"]

    return {
        "run_name": run_dir.name,
        "lr": lr,
        "max_steps_per_epoch": max_steps,
        "best_round": int(best_record["round"]),
        "best_proxy_fid": float(best_record["proxy_fid"]),
        "last_proxy_fid": float(last_record["proxy_fid"]),
        "rebound": float(last_record["proxy_fid"]) - float(best_record["proxy_fid"]),
        "tail_mean_proxy_fid": float(sum(tail) / len(tail)),
        "tail_std_proxy_fid": float(statistics.pstdev(tail)),
        "last_avg_client_loss": float(last_record["avg_client_loss"]),
    }


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    base_cfg = load_yaml(args.base_config)

    output_root = ensure_dir(args.output_root)
    generated_cfg_dir = ensure_dir(output_root / "generated_configs")
    logs_dir = ensure_dir(output_root / "logs")

    records: list[dict] = []

    for lr, max_steps in product(args.lrs, args.max_steps):
        run_name = build_run_name(lr, max_steps)
        run_dir = output_root / run_name
        config_path = generated_cfg_dir / f"{run_name}.yaml"
        log_path = logs_dir / f"{run_name}.log"
        final_metrics_path = run_dir / "metrics" / "final_metrics.json"

        cfg = deepcopy(base_cfg)
        cfg["training"]["lr"] = float(lr)
        cfg["federated"]["clients_per_round"] = 4
        cfg["federated"]["max_steps_per_epoch"] = int(max_steps)
        cfg["system"]["output_dir"] = str(run_dir).replace("\\", "/")
        cfg["system"]["save_every_round"] = False
        cfg["system"]["save_final_samples"] = False
        write_yaml(cfg, config_path)

        if args.force or not final_metrics_path.exists():
            start_time = time.perf_counter()
            print(f"[RUN] {run_name} -> lr={lr}, max_steps_per_epoch={max_steps}")
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    [sys.executable, "run_federated.py", "--config", str(config_path)],
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            duration = time.perf_counter() - start_time
            if result.returncode != 0:
                raise RuntimeError(f"Sweep run failed for {run_name}. See {log_path}.")
            print(f"[DONE] {run_name} in {duration:.1f}s")
        else:
            print(f"[SKIP] {run_name} already has final metrics")

        record = summarize_run(run_dir, lr, max_steps, args.tail_window)
        records.append(record)

    records.sort(key=lambda item: (item["rebound"], item["tail_std_proxy_fid"], item["last_proxy_fid"]))

    summary = {
        "base_config": str(Path(args.base_config).resolve()),
        "tail_window": int(args.tail_window),
        "records": records,
        "best_stability_candidate": min(
            records,
            key=lambda item: (
                item["rebound"],
                item["tail_std_proxy_fid"],
                item["last_proxy_fid"],
                item["best_proxy_fid"],
            ),
        ),
        "best_proxy_fid_candidate": min(records, key=lambda item: item["best_proxy_fid"]),
    }

    save_json(summary, output_root / "summary.json")
    write_csv(records, output_root / "summary.csv")

    print("\nTop stability candidate:")
    print(json.dumps(summary["best_stability_candidate"], indent=2, ensure_ascii=False))
    print("\nTop best-proxy-fid candidate:")
    print(json.dumps(summary["best_proxy_fid_candidate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
