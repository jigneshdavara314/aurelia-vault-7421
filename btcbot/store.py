from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config
from .errors import StoreError

SCHEMA_VERSION = 1


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def conn_ctx(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    c = _connect(db_path)
    try:
        yield c
    finally:
        c.close()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,
        strategy TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        entry_ts INTEGER NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        size_usd REAL NOT NULL,
        tp_price REAL NOT NULL,
        sl_price REAL NOT NULL,
        horizon_bars INTEGER NOT NULL,
        timeout_ts INTEGER NOT NULL,
        pred_p_up REAL,
        edge REAL NOT NULL,
        estimator TEXT NOT NULL,
        regime TEXT,
        reason TEXT NOT NULL,
        fee_bps_assumed INTEGER NOT NULL,
        slippage_bps_assumed INTEGER NOT NULL,
        status TEXT NOT NULL,
        exit_ts INTEGER,
        exit_price REAL,
        exit_reason TEXT,
        pnl_usd REAL,
        fee_usd REAL,
        slippage_usd REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy, status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades(entry_ts)",
    """
    CREATE TABLE IF NOT EXISTS bankroll (
        id INTEGER PRIMARY KEY,
        strategy TEXT,
        mode TEXT NOT NULL,
        balance REAL NOT NULL,
        initial_deposit REAL NOT NULL,
        peak_equity REAL NOT NULL,
        open_exposure REAL NOT NULL DEFAULT 0,
        drawdown_halted INTEGER NOT NULL DEFAULT 0,
        last_updated INTEGER NOT NULL,
        UNIQUE(strategy, mode)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bankroll_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bankroll_id INTEGER NOT NULL,
        ts INTEGER NOT NULL,
        delta REAL NOT NULL,
        balance_after REAL NOT NULL,
        note TEXT,
        trade_id INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_snapshots (
        day TEXT PRIMARY KEY,
        paper_equity REAL NOT NULL,
        live_equity REAL,
        open_count INTEGER NOT NULL,
        trades_opened_today INTEGER NOT NULL,
        trades_closed_today INTEGER NOT NULL,
        pnl_today REAL NOT NULL,
        peak_equity REAL NOT NULL,
        drawdown_pct REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gate_failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        symbol TEXT NOT NULL,
        gate TEXT NOT NULL,
        reason TEXT NOT NULL,
        size_usd REAL,
        pred_p_up REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signal_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        symbol TEXT NOT NULL,
        outcome TEXT NOT NULL,
        reason TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signal_events_ts ON signal_events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_signal_events_strategy ON signal_events(strategy, ts)",
    "CREATE INDEX IF NOT EXISTS idx_gate_failures_ts ON gate_failures(ts)",
]


def init_db(db_path: Path | None = None) -> None:
    with conn_ctx(db_path) as conn, tx(conn):
        for stmt in SCHEMA:
            conn.execute(stmt)
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        if row is None or row["v"] is None:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, config.time_now_ms()),
            )


def record_trade(signal_dict: dict, fill: dict, conn: sqlite3.Connection | None = None) -> int:
    cur = conn or _connect()
    own = conn is None
    try:
        if own:
            cur.execute("BEGIN IMMEDIATE")
        c = cur.execute(
            """
            INSERT INTO trades (
                mode, strategy, symbol, timeframe, entry_ts, side,
                entry_price, size_usd, tp_price, sl_price, horizon_bars, timeout_ts,
                pred_p_up, edge, estimator, regime, reason,
                fee_bps_assumed, slippage_bps_assumed, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (
                fill["mode"],
                signal_dict["strategy"],
                signal_dict["symbol"],
                signal_dict["timeframe"],
                fill["ts"],
                signal_dict["side"],
                fill["fill_price"],
                fill["fill_size_usd"],
                signal_dict["tp_price"],
                signal_dict["sl_price"],
                signal_dict["horizon_bars"],
                signal_dict["timeout_ts"],
                signal_dict.get("pred_p_up"),
                signal_dict["edge"],
                signal_dict["estimator"],
                signal_dict.get("regime"),
                signal_dict["reason"],
                fill["fee_bps_assumed"],
                fill["slippage_bps_assumed"],
            ),
        )
        trade_id = c.lastrowid
        if own:
            cur.execute("COMMIT")
        return int(trade_id)
    except Exception:
        if own:
            cur.execute("ROLLBACK")
        raise
    finally:
        if own:
            cur.close()


def settle_and_credit(
    trade_id: int,
    exit_ts: int,
    exit_price: float,
    exit_reason: str,
    fee_usd: float,
    slippage_usd: float,
    conn: sqlite3.Connection | None = None,
) -> tuple[float, float]:
    """Atomic settle. Returns (pnl_usd, new_balance)."""
    cur = conn or _connect()
    own = conn is None
    try:
        if own:
            cur.execute("BEGIN IMMEDIATE")
        row = cur.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"trade {trade_id} not found")
        if row["status"] != "OPEN":
            raise StoreError(f"trade {trade_id} already settled ({row['status']})")

        side = row["side"]
        entry = row["entry_price"]
        size = row["size_usd"]
        qty = size / entry if entry > 0 else 0.0
        if side == "LONG":
            gross = qty * (exit_price - entry)
        else:
            gross = qty * (entry - exit_price)
        pnl = gross - fee_usd - slippage_usd

        if exit_reason == "VOID":
            status = "VOID"
        elif exit_reason == "TIMEOUT":
            status = "TIMEOUT"
        else:
            status = "WON" if pnl > 0 else "LOST"

        cur.execute(
            """
            UPDATE trades SET status = ?, exit_ts = ?, exit_price = ?,
                exit_reason = ?, pnl_usd = ?, fee_usd = ?, slippage_usd = ?
            WHERE id = ?
            """,
            (status, exit_ts, exit_price, exit_reason, pnl, fee_usd, slippage_usd, trade_id),
        )

        bk = _get_bankroll_row(cur, row["strategy"], row["mode"])
        payout = size + pnl
        new_balance = bk["balance"] + payout
        new_peak = max(bk["peak_equity"], new_balance)
        cur.execute(
            """
            UPDATE bankroll SET balance = ?, peak_equity = ?, last_updated = ?
            WHERE id = ?
            """,
            (new_balance, new_peak, exit_ts, bk["id"]),
        )
        cur.execute(
            """
            INSERT INTO bankroll_log (bankroll_id, ts, delta, balance_after, note, trade_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bk["id"], exit_ts, payout, new_balance, f"settle:{exit_reason}", trade_id),
        )
        if own:
            cur.execute("COMMIT")
        return float(pnl), float(new_balance)
    except Exception:
        if own:
            cur.execute("ROLLBACK")
        raise
    finally:
        if own:
            cur.close()


def _get_bankroll_row(
    conn: sqlite3.Connection, strategy: str | None, mode: str
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM bankroll WHERE strategy IS ? AND mode = ?",
        (strategy, mode),
    ).fetchone()
    if row is None:
        raise StoreError(f"no bankroll row for strategy={strategy} mode={mode}")
    return row


def open_positions(strategy: str | None = None) -> list[dict[str, Any]]:
    with conn_ctx() as c:
        if strategy is None:
            rows = c.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_ts"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' AND strategy = ? ORDER BY entry_ts",
                (strategy,),
            ).fetchall()
        return [dict(r) for r in rows]


def already_open(symbol: str, entry_bar_ts: int, strategy: str) -> bool:
    with conn_ctx() as c:
        row = c.execute(
            """
            SELECT 1 FROM trades
            WHERE status = 'OPEN' AND symbol = ? AND entry_ts = ? AND strategy = ?
            """,
            (symbol, entry_bar_ts, strategy),
        ).fetchone()
        return row is not None


def open_position_count(strategy: str | None = None) -> int:
    with conn_ctx() as c:
        if strategy is None:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM trades WHERE status = 'OPEN'"
            ).fetchone()
        else:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM trades WHERE status = 'OPEN' AND strategy = ?",
                (strategy,),
            ).fetchone()
        return int(row["n"])


def open_count_for_symbol(symbol: str) -> int:
    with conn_ctx() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE status = 'OPEN' AND symbol = ?",
            (symbol,),
        ).fetchone()
        return int(row["n"])


def staked_today(strategy: str | None, day_start_ms: int) -> float:
    with conn_ctx() as c:
        if strategy is None:
            row = c.execute(
                "SELECT COALESCE(SUM(size_usd), 0) AS s FROM trades WHERE entry_ts >= ?",
                (day_start_ms,),
            ).fetchone()
        else:
            row = c.execute(
                """
                SELECT COALESCE(SUM(size_usd), 0) AS s FROM trades
                WHERE entry_ts >= ? AND strategy = ?
                """,
                (day_start_ms, strategy),
            ).fetchone()
        return float(row["s"])


def performance_summary(strategy: str | None = None) -> dict[str, Any]:
    with conn_ctx() as c:
        if strategy is None:
            rows = c.execute(
                """
                SELECT status, COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl
                FROM trades GROUP BY status
                """
            ).fetchall()
        else:
            rows = c.execute(
                """
                SELECT status, COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl
                FROM trades WHERE strategy = ? GROUP BY status
                """,
                (strategy,),
            ).fetchall()
    summary: dict[str, Any] = {"won": 0, "lost": 0, "timeout": 0, "open": 0, "pnl": 0.0}
    for r in rows:
        key = r["status"].lower()
        summary[key] = summary.get(key, 0) + int(r["n"])
        summary["pnl"] += float(r["pnl"] or 0)
    n_closed = summary["won"] + summary["lost"] + summary["timeout"]
    summary["n_closed"] = n_closed
    summary["win_rate"] = summary["won"] / n_closed if n_closed else 0.0
    return summary


def save_daily_snapshot(day_utc: str, now_ts: int) -> None:
    from . import bankroll as bk
    with conn_ctx() as c:
        agg = bk.aggregate_summary(mode="PAPER")
        equity = agg.get("total_equity", 0.0)
        peak = agg.get("peak_equity", 0.0)
        open_n = open_position_count()
        day_start = _day_start_ms(day_utc)
        day_end = day_start + 86_400_000
        opened = c.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE entry_ts >= ? AND entry_ts < ?",
            (day_start, day_end),
        ).fetchone()["n"]
        closed_row = c.execute(
            """
            SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl
            FROM trades WHERE exit_ts >= ? AND exit_ts < ?
            """,
            (day_start, day_end),
        ).fetchone()
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        c.execute(
            """
            INSERT OR REPLACE INTO daily_snapshots (
                day, paper_equity, live_equity, open_count,
                trades_opened_today, trades_closed_today, pnl_today,
                peak_equity, drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                day_utc, equity, None, open_n,
                int(opened), int(closed_row["n"]), float(closed_row["pnl"]),
                peak, drawdown,
            ),
        )


def _day_start_ms(day_utc: str) -> int:
    import datetime as _dt
    d = _dt.datetime.strptime(day_utc, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


def record_gate_failure(
    ts: int, strategy: str, symbol: str, gate: str, reason: str,
    size_usd: float | None = None, pred_p_up: float | None = None,
) -> None:
    with conn_ctx() as c, tx(c):
        c.execute(
            """
            INSERT INTO gate_failures (ts, strategy, symbol, gate, reason, size_usd, pred_p_up)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, strategy, symbol, gate, reason, size_usd, pred_p_up),
        )


def record_signal_event(
    ts: int, strategy: str, symbol: str, outcome: str, reason: str = "",
) -> None:
    """outcome: 'evaluated_no_signal' | 'signal_emitted' | 'signal_filled'"""
    with conn_ctx() as c, tx(c):
        c.execute(
            "INSERT INTO signal_events (ts, strategy, symbol, outcome, reason) VALUES (?, ?, ?, ?, ?)",
            (ts, strategy, symbol, outcome, reason),
        )


def signal_summary(since_ms: int, strategy: str | None = None) -> list[dict[str, Any]]:
    """Per-strategy signal-emission counts since `since_ms`."""
    with conn_ctx() as c:
        if strategy is None:
            rows = c.execute(
                """
                SELECT strategy,
                       SUM(CASE WHEN outcome='evaluated_no_signal' THEN 1 ELSE 0 END) AS n_none,
                       SUM(CASE WHEN outcome='signal_emitted'      THEN 1 ELSE 0 END) AS n_emit,
                       SUM(CASE WHEN outcome='signal_filled'       THEN 1 ELSE 0 END) AS n_fill
                FROM signal_events WHERE ts >= ?
                GROUP BY strategy ORDER BY strategy
                """, (int(since_ms),),
            ).fetchall()
        else:
            rows = c.execute(
                """
                SELECT strategy,
                       SUM(CASE WHEN outcome='evaluated_no_signal' THEN 1 ELSE 0 END) AS n_none,
                       SUM(CASE WHEN outcome='signal_emitted'      THEN 1 ELSE 0 END) AS n_emit,
                       SUM(CASE WHEN outcome='signal_filled'       THEN 1 ELSE 0 END) AS n_fill
                FROM signal_events WHERE ts >= ? AND strategy = ?
                GROUP BY strategy
                """, (int(since_ms), strategy),
            ).fetchall()
        return [dict(r) for r in rows]


def query_trades(filters: dict, limit: int = 500) -> list[dict[str, Any]]:
    where = []
    args: list[Any] = []
    for k, v in filters.items():
        if k == "since_ts":
            where.append("entry_ts >= ?")
            args.append(int(v))
        elif k == "strategy":
            where.append("strategy = ?")
            args.append(v)
        elif k == "status":
            where.append("status = ?")
            args.append(v)
        elif k == "mode":
            where.append("mode = ?")
            args.append(v)
    sql = "SELECT * FROM trades"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY entry_ts DESC LIMIT ?"
    args.append(int(limit))
    with conn_ctx() as c:
        rows = c.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def get_trade(trade_id: int) -> dict[str, Any] | None:
    with conn_ctx() as c:
        row = c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row) if row else None
