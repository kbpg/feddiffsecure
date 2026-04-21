from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ESTIMATES_PATH = ROOT / "outputs" / "paper_assets" / "runtime_estimates.json"


def load_runtime_estimates() -> dict:
    if not RUNTIME_ESTIMATES_PATH.exists():
        return {}
    return json.loads(RUNTIME_ESTIMATES_PATH.read_text(encoding="utf-8"))


def build_stl10_eta_window(live_status: dict | None) -> dict:
    if not live_status or not live_status.get("latest_round"):
        return {
            "status": "warming_up",
            "message": "STL10 还处在早期轮次，等更多轮次完成后 ETA 会更稳定。",
        }

    latest_round = int(live_status["latest_round"])
    if latest_round < 3:
        return {
            "status": "provisional",
            "message": "目前只积累了很少的轮次，当前 ETA 仍然是阶段性估计。",
            "eta_window_hours": [22.0, 28.0],
            "point_estimate_hours": 25.1,
        }

    point = live_status.get("eta_hours")
    return {
        "status": "steady",
        "message": "当前 ETA 基于最近完成轮次的稳健估计。",
        "eta_window_hours": [point * 0.9, point * 1.1] if point else None,
        "point_estimate_hours": point,
    }
