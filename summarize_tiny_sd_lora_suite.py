from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _flatten_run(run_dir: Path) -> dict[str, Any]:
    run_summary = _read_json(run_dir / "run_summary.json") or {}
    final_metrics = _read_json(run_dir / "metrics" / "final_metrics.json") or {}
    round_metrics = _read_json(run_dir / "metrics" / "round_metrics.json") or []
    audit_summary = _read_json(run_dir / "audit_summary.json") or {}
    last_round = final_metrics.get("last_round") or (round_metrics[-1] if round_metrics else {})
    best_round = final_metrics.get("best_loss_round") or {}
    payload_means = [row.get("payload_bytes_compressed_mean") for row in round_metrics if row.get("payload_bytes_compressed_mean") is not None]
    skipped_steps = [row.get("skipped_steps", 0) for row in round_metrics]

    return {
        "run_id": run_dir.name,
        "dataset": run_summary.get("dataset"),
        "resolution": run_summary.get("resolution"),
        "num_clients": run_summary.get("num_clients"),
        "clients_per_round": run_summary.get("clients_per_round"),
        "rounds": run_summary.get("rounds"),
        "lora_rank": run_summary.get("lora_rank"),
        "lora_alpha": run_summary.get("lora_alpha"),
        "lora_numel": run_summary.get("lora_numel"),
        "audit_enabled": run_summary.get("audit_enabled"),
        "best_loss": final_metrics.get("best_loss"),
        "best_loss_round": best_round.get("round"),
        "last_loss": last_round.get("avg_client_loss"),
        "last_skipped_steps": last_round.get("skipped_steps"),
        "total_skipped_steps": int(sum(int(x or 0) for x in skipped_steps)),
        "audit_records": audit_summary.get("records"),
        "audit_verified": audit_summary.get("verified"),
        "payload_bytes_compressed_mean_over_rounds": round(
            sum(float(x) for x in payload_means) / max(len(payload_means), 1), 3
        ) if payload_means else None,
        "global_lora_final_path": final_metrics.get("global_lora_final_path"),
        "sample_final": str((run_dir / "samples" / "final.png").resolve()) if (run_dir / "samples" / "final.png").exists() else "",
        "sample_best_loss": str((run_dir / "samples" / "best_loss.png").resolve()) if (run_dir / "samples" / "best_loss.png").exists() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tiny-sd LoRA experiment runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to summarize.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for summary csv/json.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_run(Path(run_dir)) for run_dir in args.run_dirs]

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
