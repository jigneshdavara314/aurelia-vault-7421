from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config, store
from .backtest import wilson_interval

TIER_MULT = {"trial": 0.25, "exploratory": 0.5, "confirmed": 1.0, "disabled": 0.0}
PROMOTE_WLB_BAR = 0.50  # Wilson lower bound on win rate
DEMOTE_WLB_BAR = 0.45
TRIAL_TO_EXPLORATORY_DAYS = 2  # was 5; at ~3 trades/day we need faster cycles
EXPLORATORY_TO_CONFIRMED_DAYS = 4  # was 10
# Reduced n thresholds: at ~1 trade/cell/day, n>=20 took ~100 days; n>=8 takes 8d
PROMOTE_N_MIN = 8   # was 20; trial -> exploratory requires this n in evaluation window
DEMOTE_ROLLING_N = 12  # was 30; demote on rolling-12 evidence
DISABLE_N_BAR = 12  # was 30
# Cell granularity: stop fanning out across regime (collapses 8 cells -> 3-6)
USE_REGIME_IN_CELL_KEY = False


@dataclass
class CellState:
    strategy: str
    regime: str
    side: str
    tier: str = "trial"
    days_in_tier: int = 0
    last_evaluated_at: int = 0
    last_changed_at: int = 0
    consecutive_promote_days: int = 0
    rolling_win_rate: float = 0.0
    rolling_n: int = 0
    rolling_wilson_lower: float = 0.0
    rolling_wilson_upper: float = 0.0
    history: list[dict] = field(default_factory=list)


def _key(c: CellState) -> str:
    if USE_REGIME_IN_CELL_KEY:
        return f"{c.strategy}|{c.regime}|{c.side}"
    # Collapsed: strategy|side only. Concentrates samples so n grows ~3x faster.
    return f"{c.strategy}|*|{c.side}"


def _path() -> Path:
    return config.STRATEGY_STATE_PATH


def _log_path() -> Path:
    return config.SELF_IMPROVE_LOG_PATH


def _load() -> dict[str, CellState]:
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, CellState] = {}
    for k, v in raw.items():
        try:
            out[k] = CellState(**v)
        except TypeError:
            continue
    return out


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save(state: dict[str, CellState]) -> None:
    payload = json.dumps({k: asdict(v) for k, v in state.items()}, indent=2)
    _atomic_write(_path(), payload)


def _append_log(row: dict[str, Any]) -> None:
    p = _log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _recent_trades(strategy: str, regime: str, side: str, n: int) -> list[dict]:
    with store.conn_ctx() as c:
        if USE_REGIME_IN_CELL_KEY:
            rows = c.execute(
                """
                SELECT * FROM trades
                WHERE strategy = ? AND regime = ? AND side = ?
                  AND status IN ('WON','LOST','TIMEOUT')
                ORDER BY exit_ts DESC LIMIT ?
                """,
                (strategy, regime, side, int(n)),
            ).fetchall()
        else:
            # Regime-agnostic aggregation: collapse across regimes for faster n growth.
            rows = c.execute(
                """
                SELECT * FROM trades
                WHERE strategy = ? AND side = ?
                  AND status IN ('WON','LOST','TIMEOUT')
                ORDER BY exit_ts DESC LIMIT ?
                """,
                (strategy, side, int(n)),
            ).fetchall()
    return [dict(r) for r in rows]


def _wins(trades: list[dict]) -> int:
    return sum(1 for t in trades if t.get("exit_reason") == "TP")


def _promote(cs: CellState, now_ts: int, reason: str) -> None:
    prev = cs.tier
    if cs.tier == "trial":
        cs.tier = "exploratory"
    elif cs.tier == "exploratory":
        cs.tier = "confirmed"
    elif cs.tier == "disabled":
        cs.tier = "trial"
    if prev != cs.tier:
        cs.days_in_tier = 0
        cs.consecutive_promote_days = 0
        cs.last_changed_at = now_ts
        cs.history.append({"ts": now_ts, "from": prev, "to": cs.tier, "reason": reason})
        _append_log({"ts": now_ts, "cell": _key(cs), "from": prev, "to": cs.tier, "reason": reason})


def _demote(cs: CellState, now_ts: int, reason: str, to: str | None = None) -> None:
    prev = cs.tier
    if to is None:
        if cs.tier == "confirmed":
            cs.tier = "exploratory"
        elif cs.tier == "exploratory":
            cs.tier = "trial"
        elif cs.tier == "trial":
            cs.tier = "disabled"
    else:
        cs.tier = to
    if prev != cs.tier:
        cs.days_in_tier = 0
        cs.consecutive_promote_days = 0
        cs.last_changed_at = now_ts
        cs.history.append({"ts": now_ts, "from": prev, "to": cs.tier, "reason": reason})
        _append_log({"ts": now_ts, "cell": _key(cs), "from": prev, "to": cs.tier, "reason": reason})


def run(now_ts: int) -> dict[str, Any]:
    state = _load()
    summary: dict[str, Any] = {"evaluated": 0, "promoted": 0, "demoted": 0, "disabled": 0}
    seen_cells: set[str] = set()

    with store.conn_ctx() as c:
        if USE_REGIME_IN_CELL_KEY:
            rows = c.execute(
                """
                SELECT DISTINCT strategy, regime, side FROM trades
                WHERE status IN ('WON','LOST','TIMEOUT')
                """
            ).fetchall()
        else:
            rows = c.execute(
                """
                SELECT DISTINCT strategy, '*' AS regime, side FROM trades
                WHERE status IN ('WON','LOST','TIMEOUT')
                """
            ).fetchall()

    for r in rows:
        strategy = r["strategy"]
        regime = r["regime"] or "*"
        side = r["side"]
        key = f"{strategy}|{regime}|{side}"
        seen_cells.add(key)
        cs = state.get(key) or CellState(strategy=strategy, regime=regime, side=side)
        recent = _recent_trades(strategy, regime, side, max(DEMOTE_ROLLING_N, 30))
        n = len(recent)
        wins = _wins(recent)
        wlb, wub = wilson_interval(wins, n)
        cs.rolling_n = n
        cs.rolling_win_rate = wins / n if n else 0.0
        cs.rolling_wilson_lower = wlb
        cs.rolling_wilson_upper = wub
        if cs.last_evaluated_at and (now_ts - cs.last_evaluated_at) >= 86_400_000:
            cs.days_in_tier += 1
        cs.last_evaluated_at = now_ts

        if cs.tier == "disabled":
            state[key] = cs
            continue

        if wlb > PROMOTE_WLB_BAR and n >= PROMOTE_N_MIN:
            cs.consecutive_promote_days += 1
        else:
            cs.consecutive_promote_days = 0

        promoted = False
        if cs.tier == "trial" and cs.consecutive_promote_days >= TRIAL_TO_EXPLORATORY_DAYS:
            _promote(cs, now_ts, f"trial→exploratory wlb={wlb:.3f} n={n}")
            summary["promoted"] += 1
            promoted = True
        elif cs.tier == "exploratory" and cs.consecutive_promote_days >= EXPLORATORY_TO_CONFIRMED_DAYS:
            _promote(cs, now_ts, f"exploratory→confirmed wlb={wlb:.3f} n={n}")
            summary["promoted"] += 1
            promoted = True

        if not promoted and n >= DEMOTE_ROLLING_N and wlb < DEMOTE_WLB_BAR:
            _demote(cs, now_ts, f"wlb={wlb:.3f} < {DEMOTE_WLB_BAR}")
            summary["demoted"] += 1
            if cs.tier == "disabled":
                summary["disabled"] += 1
        if cs.tier == "trial" and n >= DISABLE_N_BAR and wub < PROMOTE_WLB_BAR:
            _demote(cs, now_ts, f"trial disabled wub={wub:.3f} < {PROMOTE_WLB_BAR}", to="disabled")
            summary["disabled"] += 1

        state[key] = cs
        summary["evaluated"] += 1

    _save(state)
    return summary


def stake_multiplier(strategy: str, regime: str, side: str) -> float:
    state = _load()
    cs = state.get(f"{strategy}|{regime}|{side}")
    if cs is None:
        return TIER_MULT["trial"]
    return TIER_MULT.get(cs.tier, 0.0)


def all_states() -> dict[str, CellState]:
    return _load()
