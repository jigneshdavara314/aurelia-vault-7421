from __future__ import annotations

from typing import Any

import pandas as pd

from . import config, store


def _resolve_one(trade: dict, candles: pd.DataFrame, tf_ms: int) -> dict | None:
    """Pure function: returns exit dict {ts, price, reason} or None if not yet resolvable."""
    if candles.empty:
        return None
    timeout_ts = trade["timeout_ts"]
    sl = float(trade["sl_price"])
    tp = float(trade["tp_price"])
    side = trade["side"]
    for row in candles.itertuples(index=False):
        ts = int(row.open_time)
        if ts <= trade["entry_ts"]:
            continue
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)
        if side == "LONG":
            sl_hit = l <= sl
            tp_hit = h >= tp
        else:
            sl_hit = h >= sl
            tp_hit = l <= tp
        if sl_hit and tp_hit:
            return {"ts": ts, "price": sl, "reason": "SL"}
        if sl_hit:
            return {"ts": ts, "price": sl, "reason": "SL"}
        if tp_hit:
            return {"ts": ts, "price": tp, "reason": "TP"}
        if ts >= timeout_ts:
            return {"ts": ts, "price": c, "reason": "TIMEOUT"}
    last_ts = int(candles["open_time"].iloc[-1])
    if last_ts >= timeout_ts:
        last_close = float(candles["close"].iloc[-1])
        return {"ts": last_ts, "price": last_close, "reason": "TIMEOUT"}
    return None


def resolve_all_open(exchange, now_ts: int | None = None) -> list[dict[str, Any]]:
    now_ts = now_ts if now_ts is not None else config.time_now_ms()
    results: list[dict] = []
    open_pos = store.open_positions()
    for trade in open_pos:
        tf = trade["timeframe"]
        tf_ms = config.timeframe_ms(tf)
        bars_since = (now_ts - trade["entry_ts"]) // tf_ms
        n = int(bars_since) + 2
        n = min(max(n, 2), 1000)
        try:
            candles = exchange.fetch_recent_candles(trade["symbol"], tf, n, drop_unclosed=True)
        except Exception as exc:
            results.append({"trade_id": trade["id"], "error": str(exc)})
            continue
        candles = candles[candles["open_time"] > trade["entry_ts"]]
        outcome = _resolve_one(trade, candles, tf_ms)
        if outcome is None:
            continue
        size = float(trade["size_usd"])
        fee_bps = int(trade["fee_bps_assumed"])
        slip_bps = int(trade["slippage_bps_assumed"])
        fee_usd = size * (fee_bps / 10_000) * 2
        slip_usd = size * (slip_bps / 10_000) * 2
        pnl, new_bal = store.settle_and_credit(
            trade["id"], outcome["ts"], outcome["price"], outcome["reason"],
            fee_usd, slip_usd,
        )
        results.append({
            "trade_id": trade["id"], "pnl": pnl, "balance": new_bal,
            "reason": outcome["reason"], "exit_price": outcome["price"],
        })
    return results


def resolve_with_candles(trade: dict, candles: pd.DataFrame, tf_ms: int) -> dict | None:
    return _resolve_one(trade, candles, tf_ms)
