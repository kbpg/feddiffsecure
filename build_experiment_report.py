from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact markdown report for diffusion experiment comparisons.")
    parser.add_argument("--comparison-dir", type=str, required=True, help="Directory produced by compare_runs.py.")
    parser.add_argument(
        "--sampling-eval-dirs",
        type=str,
        nargs="*",
        default=[],
        help="Optional directories produced by evaluate_run_sampling.py.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Optional markdown output path. Defaults to <comparison-dir>/paper_report.md.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MNIST diffusion sampling report",
        help="Markdown title.",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _relative(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _format_float(value: float | str) -> str:
    return f"{float(value):.4f}"


def _get_eval_fid(record: dict) -> float:
    if "recomputed_eval_proxy_fid" in record:
        return float(record["recomputed_eval_proxy_fid"])
    return float(record["recomputed_final_proxy_fid"])


def _has_semantic_metrics(records: list[dict]) -> bool:
    return bool(records) and "classifier_fid" in records[0]


def _load_run_summary(comparison_dir: Path, run_name: str) -> dict | None:
    summary_path = comparison_dir.parent / run_name / "run_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _format_hours(seconds: float | int | str | None) -> str:
    if seconds is None:
        return "-"
    return f"{float(seconds) / 3600.0:.2f}h"


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    output_md = Path(args.output_md) if args.output_md else comparison_dir / "paper_report.md"

    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    records = summary["runs"]
    has_semantic = _has_semantic_metrics(records)
    run_summaries = {
        record["run_name"]: _load_run_summary(comparison_dir, record["run_name"])
        for record in records
    }

    lines = [
        f"# {args.title}",
        "",
        "## Overview",
        "",
        f"![comparison_samples]({_relative(comparison_dir / 'comparison_samples.png', output_md.parent)})",
        "",
        "## Main takeaways",
        "",
        (
            f"- Best quality by best proxy FID: `{summary['best_proxy_fid_run']['run_name']}` "
            f"with `best_proxy_fid={summary['best_proxy_fid_run']['best_proxy_fid']:.4f}`."
        ),
        (
            f"- Best final-round quality: `{summary['best_final_proxy_fid_run']['run_name']}` "
            f"with `last_proxy_fid={summary['best_final_proxy_fid_run']['last_proxy_fid']:.4f}`."
        ),
        (
            f"- Best late-stage stability: `{summary['best_stability_run']['run_name']}` "
            f"with `rebound={summary['best_stability_run']['rebound']:.4f}` and "
            f"`tail_std={summary['best_stability_run']['tail_std_proxy_fid']:.4f}`."
        ),
        "",
        "## Model comparison",
        "",
    ]
    if has_semantic and "best_classifier_fid_run" in summary:
        lines.insert(
            11,
            (
                f"- Best semantic quality by classifier-FID: `{summary['best_classifier_fid_run']['run_name']}` "
                f"with `classifier_fid={summary['best_classifier_fid_run']['classifier_fid']:.4f}`."
            ),
        )
    if len(records) >= 2:
        first = records[0]
        second = records[1]
        first_summary = run_summaries.get(first["run_name"]) or {}
        second_summary = run_summaries.get(second["run_name"]) or {}
        lines.insert(
            12,
            (
                f"- Training time contrast: `{first['run_name']}` took `{_format_hours(first_summary.get('wall_clock_seconds'))}`, "
                f"`{second['run_name']}` took `{_format_hours(second_summary.get('wall_clock_seconds'))}`."
            ),
        )

    if has_semantic:
        lines.extend(
            [
                "| run | checkpoint | rounds | lr | local_steps | sample_steps | params | trainable | trainable_pct | time | best_proxy_fid | last_proxy_fid | proxy_eval_fid | classifier_fid | conf_mean | confident@0.9 | rebound | payload_bytes_raw |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| run | checkpoint | rounds | lr | local_steps | sample_steps | params | trainable | trainable_pct | time | best_proxy_fid | last_proxy_fid | recomputed_eval_proxy_fid | rebound | payload_bytes_raw |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

    for record in records:
        run_summary = run_summaries.get(record["run_name"]) or {}
        trainable_pct = 100.0 * float(record["model_params_trainable"]) / max(float(record["model_params_total"]), 1.0)
        if has_semantic:
            lines.append(
                "| "
                f"{record['run_name']} | "
                f"{record.get('checkpoint_kind_loaded', 'final')} | "
                f"{record['rounds']} | "
                f"{float(record['lr']):.0e} | "
                f"{record['max_steps_per_epoch']} | "
                f"{record['sampling_steps']} | "
                f"{record['model_params_total']} | "
                f"{record['model_params_trainable']} | "
                f"{trainable_pct:.2f}% | "
                f"{_format_hours(run_summary.get('wall_clock_seconds'))} | "
                f"{_format_float(record['best_proxy_fid'])} | "
                f"{_format_float(record['last_proxy_fid'])} | "
                f"{_format_float(_get_eval_fid(record))} | "
                f"{_format_float(record['classifier_fid'])} | "
                f"{_format_float(record['top1_confidence_mean'])} | "
                f"{_format_float(record['top1_confident_ratio'])} | "
                f"{_format_float(record['rebound'])} | "
                f"{record['payload_bytes_raw']} |"
            )
        else:
            lines.append(
                "| "
                f"{record['run_name']} | "
                f"{record.get('checkpoint_kind_loaded', 'final')} | "
                f"{record['rounds']} | "
                f"{float(record['lr']):.0e} | "
                f"{record['max_steps_per_epoch']} | "
                f"{record['sampling_steps']} | "
                f"{record['model_params_total']} | "
                f"{record['model_params_trainable']} | "
                f"{trainable_pct:.2f}% | "
                f"{_format_hours(run_summary.get('wall_clock_seconds'))} | "
                f"{_format_float(record['best_proxy_fid'])} | "
                f"{_format_float(record['last_proxy_fid'])} | "
                f"{_format_float(_get_eval_fid(record))} | "
                f"{_format_float(record['rebound'])} | "
                f"{record['payload_bytes_raw']} |"
            )

    for sampling_dir_str in args.sampling_eval_dirs:
        sampling_dir = Path(sampling_dir_str)
        summary_csv = sampling_dir / "summary.csv"
        comparison_png = sampling_dir / "sampling_steps_comparison.png"
        if not summary_csv.exists():
            continue

        rows = _load_csv(summary_csv)
        if not rows:
            continue

        best_key = "classifier_fid" if rows and "classifier_fid" in rows[0] else "proxy_fid"
        best_row = min(rows, key=lambda row: float(row[best_key]))
        lines.extend(
            [
                "",
                f"## Sampling budget study: `{sampling_dir.parent.name}`",
                "",
                f"![sampling_steps_comparison]({_relative(comparison_png, output_md.parent)})" if comparison_png.exists() else "",
                "",
                (
                    f"Best sampling budget in this sweep: `sample_steps={best_row['sample_steps']}` "
                    + (
                        f"with `classifier_fid={float(best_row['classifier_fid']):.4f}` and "
                        f"`proxy_fid={float(best_row['proxy_fid']):.4f}`."
                        if best_key == "classifier_fid"
                        else f"with `proxy_fid={float(best_row['proxy_fid']):.4f}`."
                    )
                ),
                "",
            ]
        )
        if rows and "classifier_fid" in rows[0]:
            lines.extend(
                [
                    "| sample_steps | proxy_fid | classifier_fid | conf_mean | confident@0.9 |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in rows:
                lines.append(
                    f"| {row['sample_steps']} | {_format_float(row['proxy_fid'])} | {_format_float(row['classifier_fid'])} | "
                    f"{_format_float(row['top1_confidence_mean'])} | {_format_float(row['top1_confident_ratio'])} |"
                )
        else:
            lines.extend(
                [
                    "| sample_steps | proxy_fid |",
                    "| ---: | ---: |",
                ]
            )
            for row in rows:
                lines.append(f"| {row['sample_steps']} | {_format_float(row['proxy_fid'])} |")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {output_md}")


if __name__ == "__main__":
    main()
