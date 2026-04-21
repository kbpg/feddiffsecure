from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_audit_log(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {"records": 0, "verified": 0, "failed": 0}

    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    verified = sum(int(x.get("verified", False)) for x in records)
    return {
        "records": len(records),
        "verified": verified,
        "failed": len(records) - verified,
    }
