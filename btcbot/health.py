"""Staleness watchdog. Detects silent-fail modes that look like 'working' from
cron's perspective but aren't actually producing new state.

Returns exit code 0 (healthy) or 1 (stale). Cron can `|| true` it for now but
the metrics are logged regardless and surfaced on the dashboard.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from . import config, store


def _read_jsonl(path: Path, max_lines: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
            if len(lines) > max_lines * 2:
                lines = lines[-max_lines:]
    return lines[-max_lines:]


def check() -> dict[str, Any]:
    """Returns health-check report. Caller decides what to do with `failures`."""
    now_ms = config.time_now_ms()
    today_utc = _dt.datetime.fromtimestamp(now_ms / 1000, tz=_dt.timezone.utc).date()
    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    # 1. Pattern miner: must have run today AND added new patterns recently
    pat_path = config.ROOT / "patterns.json"
    if pat_path.exists():
        try:
            pat_state = json.loads(pat_path.read_text(encoding="utf-8"))
        except Exception:
            pat_state = {}
        metrics["patterns_total"] = len(pat_state.get("patterns", {}))
        metrics["patterns_last_run_day"] = pat_state.get("last_run_day")
        metrics["patterns_new_today"] = pat_state.get("new_pattern_count_today", 0)
        metrics["patterns_cumulative_new"] = pat_state.get("cumulative_new_patterns", 0)
        metrics["patterns_day_seed"] = pat_state.get("day_seed", 0)
        # Trial/active counts
        tiers = {"trial": 0, "active": 0, "candidate": 0}
        for p in pat_state.get("patterns", {}).values():
            t = p.get("tier", "candidate")
            tiers[t] = tiers.get(t, 0) + 1
        metrics["patterns_by_tier"] = tiers
        if pat_state.get("last_run_day") != today_utc.isoformat():
            warnings.append(
                f"patterns: last_run_day={pat_state.get('last_run_day')} != today "
                f"{today_utc.isoformat()} (may run later today)"
            )
    else:
        failures.append("patterns: patterns.json missing")

    # 2. Discovery: must have run today
    disc_path = config.ROOT / "discoveries.json"
    if disc_path.exists():
        try:
            disc_state = json.loads(disc_path.read_text(encoding="utf-8"))
        except Exception:
            disc_state = {}
        metrics["discoveries_total"] = len(disc_state.get("variants", {}))
        metrics["discoveries_last_run_day"] = disc_state.get("last_run_day")
        metrics["discoveries_days_run_count"] = disc_state.get("days_run_count", 0)
        metrics["discoveries_retired"] = len(disc_state.get("retired", []))
    else:
        failures.append("discoveries: discoveries.json missing")

    # 3. Trades: at least one signal_event in last 24h
    with store.conn_ctx() as c:
        sig_24h = c.execute(
            "SELECT COUNT(*) FROM signal_events WHERE ts >= ?",
            (now_ms - 24 * 3600 * 1000,),
        ).fetchone()[0]
        emit_24h = c.execute(
            "SELECT COUNT(*) FROM signal_events WHERE ts >= ? AND outcome='signal_emitted'",
            (now_ms - 24 * 3600 * 1000,),
        ).fetchone()[0]
        trade_24h = c.execute(
            "SELECT COUNT(*) FROM trades WHERE entry_ts >= ?",
            (now_ms - 24 * 3600 * 1000,),
        ).fetchone()[0]
        # Per-strategy emission stats over last 7d
        emit_7d = c.execute(
            """
            SELECT strategy, SUM(CASE WHEN outcome='signal_emitted' THEN 1 ELSE 0 END) AS emit
            FROM signal_events WHERE ts >= ?
            GROUP BY strategy
            """,
            (now_ms - 7 * 24 * 3600 * 1000,),
        ).fetchall()
    metrics["signal_events_24h"] = sig_24h
    metrics["signals_emitted_24h"] = emit_24h
    metrics["trades_opened_24h"] = trade_24h
    metrics["emissions_by_strategy_7d"] = {r["strategy"]: r["emit"] for r in emit_7d}
    # Silently-dead strategy alarm: any strategy that has been evaluated but
    # emitted ZERO signals in 7 days is a red flag.
    for r in emit_7d:
        if r["emit"] == 0:
            warnings.append(f"strategy {r['strategy']}: 0 emissions in last 7d (suspect dormant)")

    # 4. Ladder: must have at least 1 cell with rolling_n > 0
    ladder_path = config.ROOT / "strategy_state.json"
    if ladder_path.exists():
        try:
            ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        except Exception:
            ladder = {}
        cells = len(ladder)
        eval_cells = sum(1 for c in ladder.values()
                         if (c.get("rolling_n", 0) or 0) > 0)
        evaluable = sum(1 for c in ladder.values()
                        if (c.get("rolling_n", 0) or 0) >= 8)
        metrics["ladder_cells_total"] = cells
        metrics["ladder_cells_with_trades"] = eval_cells
        metrics["ladder_cells_evaluable"] = evaluable

    # 5. Aggregate health score
    score = 10
    if not metrics.get("patterns_total"):
        score -= 3
    if metrics.get("patterns_new_today", 0) == 0 and metrics.get("patterns_day_seed", 0) > 1:
        # After day 1, every day should add at least a few new patterns via composite rotation
        warnings.append("patterns: no new pattern names added today (composite miner may be stale)")
        score -= 1
    if metrics.get("signals_emitted_24h", 0) == 0:
        warnings.append("no signals emitted in last 24h — strategies may be silently filtering all setups")
        score -= 2
    if metrics.get("ladder_cells_evaluable", 0) == 0:
        warnings.append("ladder: 0 cells have n>=8 (still accumulating samples)")
        score -= 1
    if failures:
        score -= 5

    return {
        "ts": now_ms,
        "today_utc": today_utc.isoformat(),
        "score": max(0, score),
        "max_score": 10,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
    }


def write_report(path: Path | None = None) -> Path:
    rep = check()
    out = path or (config.ROOT / "health.json")
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return out
