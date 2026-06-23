from __future__ import annotations

import sqlite3
from typing import Any

from . import config, store


def init_bankroll(
    strategy: str | None, mode: str = "PAPER", initial_deposit: float | None = None,
) -> int:
    cfg = config.load()
    deposit = initial_deposit if initial_deposit is not None else cfg.initial_deposit
    now = config.time_now_ms()
    with store.conn_ctx() as c, store.tx(c):
        existing = c.execute(
            "SELECT id FROM bankroll WHERE strategy IS ? AND mode = ?",
            (strategy, mode),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cur = c.execute(
            """
            INSERT INTO bankroll (strategy, mode, balance, initial_deposit, peak_equity,
                                  open_exposure, drawdown_halted, last_updated)
            VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (strategy, mode, deposit, deposit, deposit, now),
        )
        bid = int(cur.lastrowid)
        c.execute(
            """
            INSERT INTO bankroll_log (bankroll_id, ts, delta, balance_after, note, trade_id)
            VALUES (?, ?, ?, ?, 'init', NULL)
            """,
            (bid, now, deposit, deposit),
        )
        return bid


def _row(strategy: str | None, mode: str = "PAPER") -> sqlite3.Row | None:
    with store.conn_ctx() as c:
        return c.execute(
            "SELECT * FROM bankroll WHERE strategy IS ? AND mode = ?",
            (strategy, mode),
        ).fetchone()


def get(strategy: str | None = None, mode: str = "PAPER") -> dict[str, Any] | None:
    r = _row(strategy, mode)
    return dict(r) if r else None


def aggregate_summary(mode: str = "PAPER") -> dict[str, Any]:
    """Roll up every per-strategy bankroll into one headline view.
    The strategy=None row is bootstrap-only and excluded from the rollup."""
    with store.conn_ctx() as c:
        rows = c.execute(
            "SELECT * FROM bankroll WHERE strategy IS NOT NULL AND mode = ? "
            "AND initial_deposit >= 2",
            (mode,),
        ).fetchall()
    if not rows:
        return {"exists": False, "strategy": None, "mode": mode}
    bal = sum(float(r["balance"]) for r in rows)
    init = sum(float(r["initial_deposit"]) for r in rows)
    peak = sum(float(r["peak_equity"]) for r in rows)
    exp = open_exposure(strategy=None, mode=mode)
    halted = any(int(r["drawdown_halted"]) for r in rows)
    total = bal + exp
    profit = total - init
    rp = profit / init if init else 0.0
    dd = (peak - total) / peak if peak > 0 else 0.0
    return {
        "exists": True, "strategy": None, "mode": mode,
        "balance": round(bal, 2), "open_exposure": round(exp, 2),
        "total_equity": round(total, 2), "initial_deposit": float(init),
        "profit": round(profit, 2), "return_pct": round(rp, 4),
        "peak_equity": round(peak, 2), "drawdown_pct": round(dd, 4),
        "drawdown_halted": halted,
    }


def balance(strategy: str | None = None, mode: str = "PAPER") -> float:
    r = _row(strategy, mode)
    return float(r["balance"]) if r else 0.0


def peak_equity(strategy: str | None = None, mode: str = "PAPER") -> float:
    r = _row(strategy, mode)
    return float(r["peak_equity"]) if r else 0.0


def open_exposure(strategy: str | None = None, mode: str = "PAPER") -> float:
    """Sum of size_usd of OPEN trades; computed live so it can't drift."""
    with store.conn_ctx() as c:
        if strategy is None:
            row = c.execute(
                "SELECT COALESCE(SUM(size_usd), 0) AS s FROM trades WHERE status='OPEN' AND mode=?",
                (mode,),
            ).fetchone()
        else:
            row = c.execute(
                """
                SELECT COALESCE(SUM(size_usd), 0) AS s FROM trades
                WHERE status='OPEN' AND mode=? AND strategy=?
                """,
                (mode, strategy),
            ).fetchone()
        return float(row["s"])


def summary(strategy: str | None = None, mode: str = "PAPER") -> dict[str, Any]:
    r = _row(strategy, mode)
    if r is None:
        return {"strategy": strategy, "mode": mode, "exists": False}
    bal = float(r["balance"])
    exp = open_exposure(strategy, mode)
    peak = float(r["peak_equity"])
    total = bal + exp
    profit = total - float(r["initial_deposit"])
    rp = profit / float(r["initial_deposit"]) if r["initial_deposit"] else 0.0
    dd = (peak - total) / peak if peak > 0 else 0.0
    return {
        "exists": True,
        "strategy": strategy,
        "mode": mode,
        "balance": round(bal, 2),
        "open_exposure": round(exp, 2),
        "total_equity": round(total, 2),
        "initial_deposit": float(r["initial_deposit"]),
        "profit": round(profit, 2),
        "return_pct": round(rp, 4),
        "peak_equity": round(peak, 2),
        "drawdown_pct": round(dd, 4),
        "drawdown_halted": bool(r["drawdown_halted"]),
    }


def deduct_stake(
    strategy: str | None, stake: float, note: str,
    trade_id: int | None = None, mode: str = "PAPER",
    conn: sqlite3.Connection | None = None,
) -> float:
    now = config.time_now_ms()
    own = conn is None
    cur = conn or store._connect()
    try:
        if own:
            cur.execute("BEGIN IMMEDIATE")
        r = cur.execute(
            "SELECT * FROM bankroll WHERE strategy IS ? AND mode = ?",
            (strategy, mode),
        ).fetchone()
        if r is None:
            raise store.StoreError(f"no bankroll for strategy={strategy} mode={mode}")
        new_bal = float(r["balance"]) - float(stake)
        cur.execute(
            "UPDATE bankroll SET balance = ?, last_updated = ? WHERE id = ?",
            (new_bal, now, r["id"]),
        )
        cur.execute(
            """
            INSERT INTO bankroll_log (bankroll_id, ts, delta, balance_after, note, trade_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (r["id"], now, -float(stake), new_bal, note, trade_id),
        )
        if own:
            cur.execute("COMMIT")
        return float(new_bal)
    except Exception:
        if own:
            cur.execute("ROLLBACK")
        raise
    finally:
        if own:
            cur.close()


def can_afford(strategy: str | None, stake: float, mode: str = "PAPER") -> bool:
    return balance(strategy, mode) >= stake


def exposure_ok(
    strategy: str | None, new_stake: float, mode: str = "PAPER",
) -> bool:
    cfg = config.load()
    bal = balance(strategy, mode)
    exp = open_exposure(strategy, mode)
    if bal <= 0:
        return False
    return (exp + new_stake) <= cfg.agg_exposure_frac * (bal + exp)


def drawdown_halted(strategy: str | None = None, mode: str = "PAPER") -> bool:
    cfg = config.load()
    r = _row(strategy, mode)
    if r is None:
        return False
    if int(r["drawdown_halted"]):
        return True
    bal = float(r["balance"])
    exp = open_exposure(strategy, mode)
    peak = float(r["peak_equity"])
    total = bal + exp
    if peak <= 0:
        return False
    dd = (peak - total) / peak
    return dd >= cfg.drawdown_halt_frac


def manual_set_halt(strategy: str | None, halted: bool, mode: str = "PAPER") -> None:
    with store.conn_ctx() as c, store.tx(c):
        c.execute(
            "UPDATE bankroll SET drawdown_halted = ? WHERE strategy IS ? AND mode = ?",
            (1 if halted else 0, strategy, mode),
        )
