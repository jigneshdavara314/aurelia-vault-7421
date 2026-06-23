from __future__ import annotations

import datetime as _dt
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import backtest, config, exchange, indicators

log = logging.getLogger(__name__)

PATTERNS_PATH = config.ROOT / "patterns.json"
PATTERN_LOG_PATH = config.ROOT / "pattern_log.jsonl"

# === Honest bar for auto-enable ============================================
# A mined pattern must clear ALL of these to be promoted:
#   - n_observations >= MIN_N (statistical power)
#   - wilson_lower_bound > break_even_dir_prob + EDGE_BUFFER (edge after cost)
#   - net_pnl_per_trade > 0 after fee+slippage
#   - HOLDS_OUT: also clears bar on a held-out prior 30-day window
#   - SAME pattern passes 2 consecutive daily mining runs
MIN_N = 50
EDGE_BUFFER = 0.02
CONSECUTIVE_RUNS_TO_PROMOTE = 2
LOOKBACK_BARS = 8640  # ~30 days of 5m candles for primary window
HELD_OUT_BARS = 8640  # additional 30 days for out-of-sample validation
HORIZON_BARS = 12     # 1h forward window
COST_BPS = 20         # 10 fee + 5 spread + 5 slippage round-trip


# === Helpers ===============================================================
def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    import math
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _break_even_dir_prob(cost_bps: int = COST_BPS) -> float:
    """For a symmetric TP/SL trade (b=1), break-even win rate is 0.5 + cost/2."""
    return 0.5 + (cost_bps / 10_000) / 2


# === Pattern miners (one per family) =======================================
@dataclass
class PatternHit:
    """A pattern detected in the data. Forward window stats attached.

    The keystone is `direction`: at the bar after the pattern occurs, did
    price move up (+1) or down (-1) over HORIZON_BARS, by more than COST_BPS?
    Anything within the cost band is 'no-trade'.
    """
    name: str
    family: str
    side: str         # 'LONG' or 'SHORT' — which way the pattern says to bet
    n: int            # n observations of the pattern
    wins: int         # times the prediction was right
    win_rate: float
    wilson_lower: float
    wilson_upper: float
    mean_return_bps: float
    net_pnl: float    # cumulative P&L if we sized each at $100, after cost
    params: dict[str, Any]


def _classify_outcomes(df: pd.DataFrame, idx: np.ndarray, side: str,
                       horizon: int = HORIZON_BARS, cost_bps: int = COST_BPS) -> tuple[int, int, list[float]]:
    """Given indices where pattern fired, look forward `horizon` bars, return
    (wins, n, per-trade return in bps). 'win' means the move went `side`
    direction by more than cost."""
    close = df["close"].to_numpy()
    n = len(close)
    wins = 0
    total = 0
    rets_bps: list[float] = []
    for i in idx:
        end = i + horizon
        if end >= n:
            continue
        entry = close[i + 1]  # enter on next bar's close
        exit_ = close[end]
        ret_bps = (exit_ - entry) / entry * 10_000
        if side == "LONG":
            adjusted = ret_bps - cost_bps
            if adjusted > 0:
                wins += 1
            rets_bps.append(adjusted)
        else:
            adjusted = -ret_bps - cost_bps
            if adjusted > 0:
                wins += 1
            rets_bps.append(adjusted)
        total += 1
    return wins, total, rets_bps


# ---- family 1: run-length (N consecutive ups/downs -> what next?) ----------
def mine_run_length(df: pd.DataFrame) -> list[PatternHit]:
    sign = np.sign(df["close"].diff().fillna(0).to_numpy())
    out: list[PatternHit] = []
    for run_len in (3, 4, 5, 6, 7, 8):
        for direction in (1, -1):
            mask = np.zeros(len(df), dtype=bool)
            for i in range(run_len, len(df)):
                if all(sign[i - k] == direction for k in range(run_len)):
                    mask[i] = True
            idx = np.where(mask)[0]
            for predict_side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, predict_side)
                if n < 20:
                    continue
                wlb, wub = _wilson(wins, n)
                mean_ret = float(np.mean(rets)) if rets else 0.0
                pnl = sum(rets) / 10_000 * 100  # $100/trade
                out.append(PatternHit(
                    name=f"runlen_{run_len}{'U' if direction == 1 else 'D'}_{predict_side}",
                    family="run_length", side=predict_side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=mean_ret, net_pnl=pnl,
                    params={"run_len": run_len, "direction": int(direction)},
                ))
    return out


# ---- family 2: body-size sequences -----------------------------------------
def _body_size_class(df: pd.DataFrame) -> np.ndarray:
    bodies = (df["close"] - df["open"]).abs().to_numpy()
    ranges = (df["high"] - df["low"]).replace(0, np.nan).to_numpy()
    rel = bodies / np.where(ranges > 0, ranges, 1.0)
    cls = np.where(rel >= 0.6, 2, np.where(rel >= 0.3, 1, 0))
    cls = cls * np.sign(df["close"].diff().fillna(0).to_numpy())
    return cls.astype(int)


def mine_body_size(df: pd.DataFrame) -> list[PatternHit]:
    cls = _body_size_class(df)
    out: list[PatternHit] = []
    # 2-bar combos: (prev_class, curr_class) -> next
    for a, b in itertools.product([-2, -1, 0, 1, 2], repeat=2):
        if a == 0 and b == 0:
            continue
        mask = np.zeros(len(cls), dtype=bool)
        for i in range(2, len(cls)):
            if cls[i - 1] == a and cls[i] == b:
                mask[i] = True
        idx = np.where(mask)[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 30:
                continue
            wlb, wub = _wilson(wins, n)
            mean_ret = float(np.mean(rets)) if rets else 0.0
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"body_{a}_{b}_{side}",
                family="body_size", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=mean_ret, net_pnl=pnl,
                params={"prev_class": a, "curr_class": b},
            ))
    return out


# ---- family 3: time-of-day -------------------------------------------------
def mine_time_of_day(df: pd.DataFrame) -> list[PatternHit]:
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    hour = ts.dt.hour.to_numpy()
    out: list[PatternHit] = []
    for h in range(24):
        mask = (hour == h)
        idx = np.where(mask)[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 50:
                continue
            wlb, wub = _wilson(wins, n)
            mean_ret = float(np.mean(rets)) if rets else 0.0
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"hour_{h:02d}_{side}",
                family="time_of_day", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=mean_ret, net_pnl=pnl,
                params={"hour_utc": h},
            ))
    return out


# ---- family 4: indicator-band ----------------------------------------------
def mine_indicator_bands(df: pd.DataFrame) -> list[PatternHit]:
    work = indicators.add_indicators(df, {
        "rsi_14": {"fn": "rsi", "args": {"n": 14}},
        "z_20": {"fn": "zscore", "args": {"n": 20}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
    })
    rsi = work["rsi_14"].to_numpy()
    z = work["z_20"].to_numpy()
    atr_pct = (work["atr_14"] / work["close"]).to_numpy()
    out: list[PatternHit] = []

    rsi_bands = [(0, 30), (30, 40), (40, 60), (60, 70), (70, 100)]
    z_bands = [(-99, -2), (-2, -1), (-1, 1), (1, 2), (2, 99)]
    for (rlo, rhi), (zlo, zhi) in itertools.product(rsi_bands, z_bands):
        mask = ((rsi >= rlo) & (rsi < rhi)
                & (z >= zlo) & (z < zhi)
                & ~np.isnan(rsi) & ~np.isnan(z))
        idx = np.where(mask)[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 50:
                continue
            wlb, wub = _wilson(wins, n)
            mean_ret = float(np.mean(rets)) if rets else 0.0
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"rsi{rlo}-{rhi}_z{zlo}-{zhi}_{side}",
                family="indicator_band", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=mean_ret, net_pnl=pnl,
                params={"rsi_lo": rlo, "rsi_hi": rhi,
                        "z_lo": zlo, "z_hi": zhi},
            ))
    return out


# === Promotion logic ========================================================
def _passes_bar(hit: PatternHit) -> bool:
    be = _break_even_dir_prob()
    return (hit.n >= MIN_N
            and hit.wilson_lower > be + EDGE_BUFFER
            and hit.net_pnl > 0)


def _load_state() -> dict[str, Any]:
    if not PATTERNS_PATH.exists():
        return {"patterns": {}}
    try:
        return json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"patterns": {}}


def _save_state(state: dict[str, Any]) -> None:
    PATTERNS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _log(row: dict[str, Any]) -> None:
    with PATTERN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _maybe_promote(state: dict[str, Any], hit: PatternHit, holds_out: bool,
                   now_ts: int) -> bool:
    pat = state["patterns"].setdefault(hit.name, {
        "family": hit.family, "side": hit.side, "params": hit.params,
        "pass_streak": 0, "tier": "candidate", "first_seen": now_ts,
        "history": [],
    })
    pat["history"].append({
        "ts": now_ts, "n": hit.n, "wr": hit.win_rate,
        "wlb": hit.wilson_lower, "net_pnl": hit.net_pnl,
        "passes": _passes_bar(hit), "holds_out": holds_out,
    })
    pat["history"] = pat["history"][-30:]
    if _passes_bar(hit) and holds_out:
        pat["pass_streak"] += 1
    else:
        pat["pass_streak"] = 0

    promoted = False
    if (pat["pass_streak"] >= CONSECUTIVE_RUNS_TO_PROMOTE
            and pat["tier"] != "active"):
        pat["tier"] = "active"
        pat["promoted_at"] = now_ts
        promoted = True
        _log({"event": "promote", "pattern": hit.name, "wlb": hit.wilson_lower,
              "n": hit.n, "side": hit.side, "ts": now_ts})
    elif _passes_bar(hit):
        pat["tier"] = "trial"
    return promoted


def _activate_promoted(state: dict[str, Any]) -> int:
    """Promoted patterns are addressable as 'pattern::<name>' in settings."""
    settings_path = config.SETTINGS_PATH
    s = json.loads(settings_path.read_text(encoding="utf-8"))
    active = list(s.get("active_strategies", []))
    added = 0
    for name, pat in state["patterns"].items():
        if pat.get("tier") == "active":
            tag = f"pattern::{name}"
            if tag not in active:
                active.append(tag)
                added += 1
    if added:
        s["active_strategies"] = active
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return added


# === Main entry point =======================================================
def _fetch(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
    ex = exchange.Exchange(config.load().exchange)
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
            log.warning("fetch chunk failed: %s", exc)
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


def run(symbol: str | None = None, timeframe: str = "5m",
        n_primary: int = LOOKBACK_BARS, n_holdout: int = HELD_OUT_BARS) -> dict[str, Any]:
    cfg = config.load(force=True)
    symbol = symbol or cfg.symbol
    state = _load_state()
    now_ts = config.time_now_ms()
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_run_day") == today:
        log.info("pattern mining already ran today (%s); skipping", today)
        return {"skipped": True, "day": today}

    total_bars = n_primary + n_holdout
    log.info("patterns: fetching %d bars", total_bars)
    df = _fetch(symbol, timeframe, total_bars)
    if df.empty or len(df) < n_primary + 500:
        log.warning("patterns: not enough data (%d bars)", len(df))
        return {"error": "no_data", "bars": len(df)}

    df_holdout = df.iloc[:-n_primary].reset_index(drop=True)
    df_primary = df.iloc[-n_primary:].reset_index(drop=True)

    log.info("primary: %d bars  holdout: %d bars", len(df_primary), len(df_holdout))

    miners = {
        "run_length": mine_run_length,
        "body_size": mine_body_size,
        "time_of_day": mine_time_of_day,
        "indicator_band": mine_indicator_bands,
    }
    all_primary: list[PatternHit] = []
    all_holdout_by_name: dict[str, PatternHit] = {}

    for fname, mine_fn in miners.items():
        try:
            primary_hits = mine_fn(df_primary)
            holdout_hits = mine_fn(df_holdout)
        except Exception as exc:
            log.warning("miner %s errored: %s", fname, exc)
            continue
        all_primary.extend(primary_hits)
        for h in holdout_hits:
            all_holdout_by_name[h.name] = h

    promoted_now: list[str] = []
    for hit in all_primary:
        holdout = all_holdout_by_name.get(hit.name)
        holds_out = holdout is not None and _passes_bar(holdout)
        if _maybe_promote(state, hit, holds_out, now_ts):
            promoted_now.append(hit.name)

    state["last_run_day"] = today
    state["last_run_ts"] = now_ts
    _save_state(state)
    n_activated = _activate_promoted(state)

    passed = sum(1 for h in all_primary if _passes_bar(h))
    summary = {
        "day": today,
        "patterns_tested": len(all_primary),
        "passed_bar_primary": passed,
        "promoted_this_run": promoted_now,
        "newly_active_in_settings": n_activated,
        "total_active_patterns": sum(
            1 for p in state["patterns"].values() if p.get("tier") == "active"
        ),
    }
    _log({"event": "summary", "ts": now_ts, **summary})
    return summary


def list_patterns() -> list[dict[str, Any]]:
    state = _load_state()
    out = []
    for name, pat in state["patterns"].items():
        last = pat["history"][-1] if pat["history"] else {}
        out.append({
            "name": name, "family": pat.get("family"),
            "side": pat.get("side"), "params": pat.get("params"),
            "tier": pat.get("tier", "candidate"),
            "pass_streak": pat.get("pass_streak", 0),
            "last_n": last.get("n"), "last_wr": last.get("wr"),
            "last_wlb": last.get("wlb"), "last_net_pnl": last.get("net_pnl"),
            "last_holds_out": last.get("holds_out"),
        })
    out.sort(key=lambda r: (-(r["last_wlb"] or 0), r["name"]))
    return out


def get_active_pattern(name: str) -> dict[str, Any] | None:
    """Resolve 'pattern::<name>' to its full record so the strategy class
    can decide whether to fire."""
    state = _load_state()
    pat = state["patterns"].get(name)
    if pat is None or pat.get("tier") != "active":
        return None
    return pat
