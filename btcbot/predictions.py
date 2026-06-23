from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config

_CACHE: dict[str, dict[str, Any]] | None = None


def _path() -> Path:
    return config.PREDICTIONS_PATH


def _todo_path() -> Path:
    return config.PREDICTIONS_TODO_PATH


def load_predictions(force: bool = False) -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    out: dict[str, dict[str, Any]] = {}
    p = _path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = row.get("snapshot_id")
            if sid:
                out[sid] = row
    _CACHE = out
    return out


def get_prediction(snapshot_id: str) -> dict[str, Any] | None:
    return load_predictions().get(snapshot_id)


def append_prediction(row: dict[str, Any]) -> None:
    global _CACHE
    if "snapshot_id" not in row:
        raise ValueError("snapshot_id required")
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    if _CACHE is not None:
        _CACHE[row["snapshot_id"]] = row


def export_snapshots_todo(snapshots: list, prompt_template: str | None = None) -> Path:
    path = _todo_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in snapshots:
            row = {
                "snapshot_id": s.id if hasattr(s, "id") else s["snapshot_id"],
                "symbol": s.symbol if hasattr(s, "symbol") else s.get("symbol"),
                "timeframe": s.timeframe if hasattr(s, "timeframe") else s.get("timeframe"),
                "ts": s.ts if hasattr(s, "ts") else s.get("ts"),
                "close": s.close if hasattr(s, "close") else s.get("close"),
                "indicators": s.indicators if hasattr(s, "indicators") else s.get("indicators", {}),
                "regime": s.regime if hasattr(s, "regime") else s.get("regime"),
            }
            f.write(json.dumps(row) + "\n")
    if prompt_template:
        (path.parent / "predict_direction.md").write_text(prompt_template, encoding="utf-8")
    return path


def clear_cache() -> None:
    global _CACHE
    _CACHE = None
