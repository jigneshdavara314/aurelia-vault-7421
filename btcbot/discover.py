from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import backtest, config, exchange, strategies as strat_mod
from .backtest import wilson_interval
from .strategies.base import Strategy

log = logging.getLogger(__name__)

DISCOVERIES_PATH = config.ROOT / "discoveries.json"
DISCOVERY_LOG_PATH = config.ROOT / "discovery_log.jsonl"

# === Discovery thresholds (the honest bar) =================================
# Auto-promote variant only if:
#   - n_trades >= MIN_N
#   - wilson_lower > break_even_win_rate + EDGE_BUFFER (NET of fees + slippage)
#   - SAME variant passes 2 consecutive daily discovery runs
MIN_N = 30
EDGE_BUFFER = 0.02  # require WLB at least 2pp above the break-even WR
CONSECUTIVE_RUNS_TO_PROMOTE = 2
LOOKBACK_BARS = 8640  # ~30 days of 5m candles
MAX_VARIANTS_PER_STRATEGY = 20


@dataclass
class Variant:
    parent: str
    name: str
    params: dict[str, Any]

    @property
    def id(self) -> str:
        return f"{self.parent}::{self.name}"


@dataclass
class DiscoveryResult:
    variant_id: str
    parent: str
    params: dict[str, Any]
    n_trades: int
    win_rate: float
    wilson_lower: float
    wilson_upper: float
    net_pnl: float
    break_even_wr: float
    pass_bar: bool
    timestamp: int


# === Variant generation ====================================================
def _variants_for(name: str) -> list[Variant]:
    """Generate parameter variations for a parent strategy."""
    if name == "nsigma_fade":
        out = []
        for z, tp, sl, hor in itertools.product(
            (-1.2, -1.5, -1.8, -2.0, -2.5),
            (1.0, 1.2, 1.5, 2.0),
            (0.5, 0.7, 1.0),
            (6, 12, 24),
        ):
            out.append(Variant(
                parent=name, name=f"z{z}_tp{tp}_sl{sl}_h{hor}",
                params={"Z_THRESH": z, "TP_ATR": tp, "SL_ATR": sl, "HORIZON_BARS": hor},
            ))
        return out[:MAX_VARIANTS_PER_STRATEGY]
    if name == "breakout_donchian":
        out = []
        for tp, sl, hor in itertools.product(
            (1.5, 2.0, 2.5, 3.0),
            (0.8, 1.0, 1.5),
            (12, 24, 36),
        ):
            out.append(Variant(
                parent=name, name=f"tp{tp}_sl{sl}_h{hor}",
                params={"TP_ATR": tp, "SL_ATR": sl, "HORIZON_BARS": hor},
            ))
        return out[:MAX_VARIANTS_PER_STRATEGY]
    if name == "momentum_ema_cross":
        out = []
        for tp, sl, hor, pull in itertools.product(
            (2.0, 3.0, 4.0),
            (1.0, 1.5, 2.0),
            (24, 36, 48),
            (0.5, 1.0, 1.5),
        ):
            out.append(Variant(
                parent=name, name=f"tp{tp}_sl{sl}_h{hor}_pb{pull}",
                params={"TP_ATR": tp, "SL_ATR": sl, "HORIZON_BARS": hor, "PULLBACK_ATR": pull},
            ))
        return out[:MAX_VARIANTS_PER_STRATEGY]
    return []


def _instantiate(variant: Variant) -> Strategy:
    parent_cls = strat_mod.REGISTRY.get(variant.parent)
    if parent_cls is None:
        raise ValueError(f"unknown parent {variant.parent}")

    class _V(parent_cls):
        name = variant.id

    inst = _V()
    for k, v in variant.params.items():
        setattr(inst, k, v)
    return inst


# === Backtest data ==========================================================
def _fetch_history(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
    ex = exchange.Exchange("binance")
    chunks = []
    remaining = n_bars
    end_ts = None
    while remaining > 0:
        limit = min(1000, remaining + 1)
        try:
            raw = ex._retry(
                ex._ex.fetch_ohlcv,
                symbol, timeframe,
                None if end_ts is None else end_ts - limit * config.timeframe_ms(timeframe),
                limit,
            )
        except Exception as exc:
            log.warning("history fetch chunk failed: %s", exc)
            break
        if not raw:
            break
        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume"])
        chunks.append(df)
        end_ts = int(df["open_time"].iloc[0])
        remaining -= len(df)
        if len(df) < limit:
            break
    if not chunks:
        return pd.DataFrame()
    full = pd.concat(chunks).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    if len(full) >= 2:
        full = full.iloc[:-1].reset_index(drop=True)
    return full


# === Discovery persistence ==================================================
def _load_state() -> dict[str, Any]:
    if not DISCOVERIES_PATH.exists():
        return {"variants": {}}
    try:
        return json.loads(DISCOVERIES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"variants": {}}


def _save_state(state: dict[str, Any]) -> None:
    DISCOVERIES_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _log_event(row: dict[str, Any]) -> None:
    with DISCOVERY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# === Promotion logic ========================================================
def _maybe_promote(state: dict[str, Any], result: DiscoveryResult, now_ts: int) -> bool:
    var = state["variants"].setdefault(result.variant_id, {
        "parent": result.parent, "params": result.params,
        "pass_streak": 0, "tier": "candidate", "first_seen": now_ts, "history": [],
    })
    var["history"].append({
        "ts": now_ts, "n": result.n_trades, "wlb": result.wilson_lower,
        "wr": result.win_rate, "net": result.net_pnl,
        "be": result.break_even_wr, "pass": result.pass_bar,
    })
    var["history"] = var["history"][-30:]

    if result.pass_bar:
        var["pass_streak"] += 1
    else:
        var["pass_streak"] = 0

    promoted = False
    if (var["pass_streak"] >= CONSECUTIVE_RUNS_TO_PROMOTE
            and var["tier"] != "active"):
        var["tier"] = "active"
        var["promoted_at"] = now_ts
        promoted = True
        _log_event({"event": "promote", "variant": result.variant_id,
                    "wlb": result.wilson_lower, "be": result.break_even_wr,
                    "n": result.n_trades, "ts": now_ts})
    elif result.pass_bar:
        var["tier"] = "trial"

    return promoted


def _activate_promoted_variants(state: dict[str, Any]) -> int:
    """Append promoted variant ids to settings.json active_strategies."""
    settings_path = config.SETTINGS_PATH
    s = json.loads(settings_path.read_text(encoding="utf-8"))
    active = list(s.get("active_strategies", []))
    added = 0
    for vid, var in state["variants"].items():
        if var.get("tier") == "active" and vid not in active:
            active.append(vid)
            added += 1
    if added:
        s["active_strategies"] = active
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return added


# === Main entry point =======================================================
def run(symbol: str = "BTC/USDT", timeframe: str = "5m",
        n_bars: int = LOOKBACK_BARS) -> dict[str, Any]:
    """Run one discovery sweep. Idempotent within a day."""
    state = _load_state()
    now_ts = config.time_now_ms()
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_run_day") == today:
        log.info("discovery already ran today (%s); skipping", today)
        return {"skipped": True, "day": today}

    log.info("discovery: fetching %d bars of %s %s", n_bars, symbol, timeframe)
    df = _fetch_history(symbol, timeframe, n_bars)
    if df.empty or len(df) < 500:
        log.warning("discovery: not enough data (%d bars)", len(df))
        return {"error": "no_data", "bars": len(df)}

    gcfg = config.load(force=True)
    scfg = config.profile(gcfg.profile)
    results: list[DiscoveryResult] = []
    promoted_now: list[str] = []

    parents = ["nsigma_fade", "breakout_donchian", "momentum_ema_cross"]
    for parent in parents:
        for variant in _variants_for(parent):
            try:
                strat = _instantiate(variant)
                res = backtest.simulate(
                    strat, df, symbol, timeframe,
                    scfg, gcfg, scfg.bankroll_usd,
                    cost_bps=gcfg.paper_fee_bps + gcfg.paper_slippage_bps + 5,
                )
            except Exception as exc:
                log.warning("variant %s errored: %s", variant.id, exc)
                continue

            avg_b = 0.0
            for t in res.trades:
                risk = abs(t["entry_price"] - t["sl_price"])
                reward = abs(t["tp_price"] - t["entry_price"])
                if risk > 0:
                    avg_b += reward / risk
            avg_b = avg_b / len(res.trades) if res.trades else 0.0
            be = backtest.break_even_win_rate(avg_b, gcfg.paper_fee_bps + gcfg.paper_slippage_bps + 5)
            pass_bar = (
                res.n_trades >= MIN_N
                and res.win_rate_wilson_lower > be + EDGE_BUFFER
                and res.net_pnl > 0
            )
            dr = DiscoveryResult(
                variant_id=variant.id, parent=parent, params=variant.params,
                n_trades=res.n_trades, win_rate=res.win_rate,
                wilson_lower=res.win_rate_wilson_lower,
                wilson_upper=res.win_rate_wilson_upper,
                net_pnl=res.net_pnl, break_even_wr=be,
                pass_bar=pass_bar, timestamp=now_ts,
            )
            results.append(dr)
            if _maybe_promote(state, dr, now_ts):
                promoted_now.append(variant.id)

    state["last_run_day"] = today
    state["last_run_ts"] = now_ts
    _save_state(state)
    n_activated = _activate_promoted_variants(state)

    summary = {
        "day": today,
        "variants_tested": len(results),
        "passed_bar": sum(1 for r in results if r.pass_bar),
        "promoted_this_run": promoted_now,
        "newly_active_in_settings": n_activated,
        "total_active_variants": sum(
            1 for v in state["variants"].values() if v.get("tier") == "active"
        ),
    }
    _log_event({"event": "summary", "ts": now_ts, **summary})
    return summary


def list_variants() -> list[dict[str, Any]]:
    state = _load_state()
    out = []
    for vid, var in state["variants"].items():
        last = var["history"][-1] if var["history"] else {}
        out.append({
            "variant": vid, "parent": var.get("parent"),
            "tier": var.get("tier", "candidate"),
            "pass_streak": var.get("pass_streak", 0),
            "last_n": last.get("n"), "last_wr": last.get("wr"),
            "last_wlb": last.get("wlb"), "last_be": last.get("be"),
            "last_net": last.get("net"),
        })
    out.sort(key=lambda r: (-(r["last_wlb"] or 0), r["variant"]))
    return out
