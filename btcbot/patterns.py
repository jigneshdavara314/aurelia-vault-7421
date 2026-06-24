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
MIN_N = 30
EDGE_BUFFER = 0.005
CONSECUTIVE_RUNS_TO_PROMOTE = 2
# Mine the maximum honest data we can pull from binanceus REST without paying:
# 120 days × 288 5m bars/day = ~34560 bars per timeframe. With 4 timeframes
# (1m, 5m, 15m, 1h) we evaluate every pattern across multiple regimes.
LOOKBACK_BARS = 34560
HELD_OUT_BARS = 11520  # ~40 day out-of-sample window
HORIZON_BARS = 12      # forward window (in candles of mining timeframe)
COST_BPS = 20          # 10 fee + 5 spread + 5 slippage round-trip
TIMEFRAMES_TO_MINE = ["5m", "15m", "1h"]  # 1m too noisy for honest patterns

# trial-tier bar (looser; promoted to live with 0.25x stake)
TRIAL_MIN_N = 30
TRIAL_WLB = 0.50  # just above coinflip net of cost — surfaces near-misses


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
    """Sweep N=3..12 consecutive ups/downs. Longer streaks are rarer but
    historically have shown stronger mean-reversion edge on BTC at higher tf."""
    sign = np.sign(df["close"].diff().fillna(0).to_numpy())
    out: list[PatternHit] = []
    for run_len in (3, 4, 5, 6, 7, 8, 9, 10, 12):
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



# ---- family 5: candlestick formations --------------------------------------
def mine_candlestick(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    is_green = df["close"] > df["open"]
    is_red = df["close"] < df["open"]
    prev_green = is_green.shift(1).fillna(False)
    prev_red = is_red.shift(1).fillna(False)
    o = df["open"]
    c = df["close"]
    h = df["high"]
    l = df["low"]
    formations = {
        "doji": (body / rng < 0.10).fillna(False),
        "hammer": ((lower > 2 * body) & (upper < body) & (rng > 0)).fillna(False),
        "shooting_star": ((upper > 2 * body) & (lower < body) & (rng > 0)).fillna(False),
        "bull_engulfing": prev_red & is_green & (c > o.shift(1)) & (o < c.shift(1)),
        "bear_engulfing": prev_green & is_red & (o > c.shift(1)) & (c < o.shift(1)),
        "bull_harami": prev_red & is_green & (h < o.shift(1)) & (l > c.shift(1)),
        "bear_harami": prev_green & is_red & (h < c.shift(1)) & (l > o.shift(1)),
        "inside_bar": (h <= h.shift(1)) & (l >= l.shift(1)),
        "outside_bar": (h > h.shift(1)) & (l < l.shift(1)),
    }
    for fname, mask in formations.items():
        idx = np.where(mask.to_numpy())[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 20:
                continue
            wlb, wub = _wilson(wins, n)
            mean_ret = float(np.mean(rets)) if rets else 0.0
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"candle_{fname}_{side}", family="candlestick", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=mean_ret, net_pnl=pnl,
                params={"formation": fname},
            ))
    return out


# ---- family 6: volatility regime transition --------------------------------
def mine_vol_regime(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    ret = df["close"].pct_change()
    vol_s = ret.rolling(12, min_periods=12).std()
    vol_l = ret.rolling(96, min_periods=96).std()
    ratio = (vol_s / vol_l.replace(0, np.nan)).to_numpy()
    edges = [0.5, 0.8, 1.25, 2.0]
    buckets = np.full_like(ratio, -1, dtype=int)
    for i, r in enumerate(ratio):
        if np.isnan(r):
            continue
        buckets[i] = sum(r >= e for e in edges)
    prev = np.roll(buckets, 1)
    prev[0] = -1
    for b in range(5):
        mask = (buckets == b)
        idx = np.where(mask)[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 30:
                continue
            wlb, wub = _wilson(wins, n)
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"volreg_b{b}_{side}", family="vol_regime", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                params={"bucket": b},
            ))
    for a, b in [(0, 4), (1, 4), (4, 0), (4, 1)]:
        mask = (prev == a) & (buckets == b)
        idx = np.where(mask)[0]
        for side in ("LONG", "SHORT"):
            wins, n, rets = _classify_outcomes(df, idx, side)
            if n < 20:
                continue
            wlb, wub = _wilson(wins, n)
            pnl = sum(rets) / 10_000 * 100
            out.append(PatternHit(
                name=f"volreg_{a}to{b}_{side}", family="vol_regime", side=side,
                n=n, wins=wins, win_rate=wins / n,
                wilson_lower=wlb, wilson_upper=wub,
                mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                params={"from": a, "to": b},
            ))
    return out


# ---- family 7: volume direction spike --------------------------------------
def mine_volume_spike(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    v = df["volume"].to_numpy(dtype=float)
    mu = pd.Series(v).rolling(96, min_periods=96).mean().to_numpy()
    sd = pd.Series(v).rolling(96, min_periods=96).std(ddof=0).to_numpy()
    z = (v - mu) / np.where(sd > 0, sd, np.nan)
    bar_dir = np.sign((df["close"] - df["open"]).to_numpy())
    buckets = [(-99, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 99)]
    for (lo, hi) in buckets:
        for d in (1, -1):
            mask = (z >= lo) & (z < hi) & (bar_dir == d) & ~np.isnan(z)
            idx = np.where(mask)[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"volspike_z{lo}-{hi}_d{d}_{side}", family="volume_spike", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                    params={"z_lo": lo, "z_hi": hi, "bar_dir": int(d)},
                ))
    return out


# ---- family 8: MA-distance regime ------------------------------------------
def mine_ma_distance(df: pd.DataFrame) -> list[PatternHit]:
    """Sweep MA windows {10,20,30,50,80,100,200} with finer 9-bucket distance
    bands. Found to be the most fruitful family on initial 996-pattern scan,
    so expanding the parameter space."""
    out: list[PatternHit] = []
    edges = [-0.05, -0.025, -0.012, -0.006, -0.002, 0.002, 0.006, 0.012, 0.025, 0.05]
    for w in (10, 20, 30, 50, 80, 100, 200):
        ma = df["close"].rolling(w, min_periods=w).mean()
        dist = ((df["close"] - ma) / ma).to_numpy()
        bucket = np.full_like(dist, -1, dtype=int)
        for i, d in enumerate(dist):
            if np.isnan(d):
                continue
            bucket[i] = sum(d >= e for e in edges)
        for b in range(len(edges) + 1):
            mask = (bucket == b)
            idx = np.where(mask)[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"madist_w{w}_b{b}_{side}", family="ma_distance", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                    params={"ma_window": w, "bucket": b},
                ))
    return out


# ---- family 9: multi-timeframe slope alignment -----------------------------
def mine_mtf_slope(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    c = df["close"]
    slope_fast = (c - c.shift(1)).to_numpy()
    slope_med = (c.rolling(3, min_periods=3).mean() - c.rolling(3, min_periods=3).mean().shift(3)).to_numpy()
    slope_long = (c.rolling(12, min_periods=12).mean() - c.rolling(12, min_periods=12).mean().shift(12)).to_numpy()
    sf = np.sign(slope_fast)
    sm = np.sign(slope_med)
    sl = np.sign(slope_long)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for d in (-1, 0, 1):
                mask = (sf == a) & (sm == b) & (sl == d) & ~np.isnan(slope_long)
                idx = np.where(mask)[0]
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    pnl = sum(rets) / 10_000 * 100
                    out.append(PatternHit(
                        name=f"mtf_{a}{b}{d}_{side}", family="mtf_slope", side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                        params={"fast": a, "med": b, "long": d},
                    ))
    return out


# ---- family 10: range compression / expansion ------------------------------
def mine_range_compression(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    for w in (4, 7, 12, 24):
        rmin = tr.rolling(w, min_periods=w).min()
        rmax = tr.rolling(w, min_periods=w).max()
        for label, mask in (("NR", (tr == rmin) & rmin.notna()),
                             ("WR", (tr == rmax) & rmax.notna())):
            idx = np.where(mask.to_numpy())[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"range_{label}{w}_{side}", family="range_compression", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                    params={"window": w, "type": label},
                ))
    return out


# ---- family 11: ADX trend strength -----------------------------------------
def mine_adx(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    work = indicators.add_indicators(df, {
        "adx_14": {"fn": "adx", "args": {"n": 14}},
        "plus_di_14": {"fn": "plus_di", "args": {"n": 14}},
        "minus_di_14": {"fn": "minus_di", "args": {"n": 14}},
    })
    adx_v = work["adx_14"].to_numpy()
    di_diff = (work["plus_di_14"] - work["minus_di_14"]).to_numpy()
    adx_bands = [(0, 15), (15, 25), (25, 40), (40, 100)]
    di_bands = [(-99, -10), (-10, 10), (10, 99)]
    for (alo, ahi) in adx_bands:
        for (dlo, dhi) in di_bands:
            mask = ((adx_v >= alo) & (adx_v < ahi) & (di_diff >= dlo)
                    & (di_diff < dhi) & ~np.isnan(adx_v))
            idx = np.where(mask)[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"adx{alo}-{ahi}_di{dlo}-{dhi}_{side}", family="adx_trend", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                    params={"adx_lo": alo, "adx_hi": ahi, "di_lo": dlo, "di_hi": dhi},
                ))
    return out


# ---- family 12: swing extremum count ---------------------------------------
def mine_swing_count(df: pd.DataFrame) -> list[PatternHit]:
    out: list[PatternHit] = []
    h, l = df["high"], df["low"]
    for w in (12, 24, 48):
        new_high = (h > h.rolling(w, min_periods=w).max().shift(1)).astype(int)
        new_low = (l < l.rolling(w, min_periods=w).min().shift(1)).astype(int)
        hh_count = new_high.rolling(w, min_periods=w).sum().to_numpy()
        ll_count = new_low.rolling(w, min_periods=w).sum().to_numpy()
        for thr in (2, 3, 5):
            for label, arr in (("HH", hh_count), ("LL", ll_count)):
                mask = arr >= thr
                idx = np.where(mask)[0]
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    pnl = sum(rets) / 10_000 * 100
                    out.append(PatternHit(
                        name=f"swing_{label}{w}_t{thr}_{side}", family="swing_count", side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=float(np.mean(rets)) if rets else 0.0, net_pnl=pnl,
                        params={"window": w, "kind": label, "threshold": thr},
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


def _coerce(o):
    """Make numpy types JSON serializable."""
    if isinstance(o, dict):
        return {k: _coerce(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _save_state(state: dict[str, Any]) -> None:
    PATTERNS_PATH.write_text(json.dumps(_coerce(state), indent=2), encoding="utf-8")


def _log(row: dict[str, Any]) -> None:
    with PATTERN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_coerce(row)) + "\n")


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




# ---- new family: variance_of_variance (auto-implemented) ----
def mine_variance_of_variance(df: pd.DataFrame) -> list[PatternHit]:
    """Heteroskedasticity clustering: rolling std-of-std of returns.

    Crosses vol-of-vol z-score with absolute vol regime to detect regimes
    where volatility itself is volatile vs steady, independently of vol level.
    Orthogonal to vol_regime (which only captures the 1st moment of vol).
    """
    out: list[PatternHit] = []
    ret = df["close"].pct_change()
    n_bars = len(df)

    for short_w in (10, 20):
        short_vol = ret.rolling(short_w).std()
        for vov_w in (30, 60, 120):
            vov = short_vol.rolling(vov_w).std()
            vov_mean = vov.rolling(vov_w * 2).mean()
            vov_std = vov.rolling(vov_w * 2).std() + 1e-12
            vov_z = (vov - vov_mean) / vov_std

            long_vol = ret.rolling(vov_w).std()
            vol_level_z = (short_vol - long_vol) / (long_vol + 1e-12)

            vov_z_arr = vov_z.to_numpy()
            level_arr = vol_level_z.to_numpy()

            # bucket: low z<-0.5, mid -0.5..0.5, high z>0.5
            vov_band_arr = np.where(
                vov_z_arr < -0.5, 0,
                np.where(vov_z_arr > 0.5, 2, 1),
            )
            # for level use same -0.5/+0.5 thresholds on its z-like measure
            level_band_arr = np.where(
                level_arr < -0.5, 0,
                np.where(level_arr > 0.5, 2, 1),
            )

            valid = (
                ~np.isnan(vov_z_arr)
                & ~np.isnan(level_arr)
                & ~np.isnan(short_vol.to_numpy())
            )

            band_names = {0: "low", 1: "mid", 2: "high"}
            for vov_b in (0, 1, 2):
                for lvl_b in (0, 1, 2):
                    mask = valid & (vov_band_arr == vov_b) & (level_band_arr == lvl_b)
                    idx = np.where(mask)[0]
                    if len(idx) < 30:
                        continue
                    for side in ("LONG", "SHORT"):
                        wins, n, rets = _classify_outcomes(df, idx, side)
                        if n < 30:
                            continue
                        wlb, wub = _wilson(wins, n)
                        mean_ret = float(np.mean(rets)) if rets else 0.0
                        pnl = sum(rets) / 10_000 * 100
                        out.append(PatternHit(
                            name=f"variance_of_variance_sw{short_w}_vw{vov_w}_vov-{band_names[vov_b]}_lvl-{band_names[lvl_b]}_{side}",
                            family="variance_of_variance", side=side,
                            n=n, wins=wins, win_rate=wins / n,
                            wilson_lower=wlb, wilson_upper=wub,
                            mean_return_bps=mean_ret, net_pnl=pnl,
                            params={
                                "short_w": short_w,
                                "vov_w": vov_w,
                                "vov_band": band_names[vov_b],
                                "level_band": band_names[lvl_b],
                                "side": side,
                            },
                        ))
    return out




# ---- new family: mean_reversion_halflife (auto-implemented) ----
def mine_mean_reversion_halflife(df: pd.DataFrame) -> list[PatternHit]:
    """Fit rolling AR(1) on log-price, extract implied half-life of mean
    reversion (-ln2/ln(phi)), and bucket into bands. Captures statistical
    pull-back speed, distinct from current deviation magnitude."""
    out: list[PatternHit] = []
    if len(df) < 300:
        return out

    lp = np.log(df["close"].to_numpy())

    def ar1_phi(arr: np.ndarray) -> float:
        x = arr[:-1] - arr[:-1].mean()
        y = arr[1:] - arr[1:].mean()
        denom = (x ** 2).sum()
        if denom < 1e-12:
            return np.nan
        return (x * y).sum() / denom

    lp_series = pd.Series(lp)

    for win in (60, 120, 240):
        if len(df) < win + 30:
            continue
        phi = lp_series.rolling(win).apply(ar1_phi, raw=True).to_numpy()
        # half-life: -ln(2)/ln(phi) for 0 < phi < 1
        with np.errstate(invalid="ignore", divide="ignore"):
            phi_clip = np.clip(phi, 1e-6, 0.9999)
            hl = np.where(
                (phi > 0) & (phi < 1),
                -np.log(2) / np.log(phi_clip),
                np.nan,
            )

        # bucket assignment
        band = np.full(len(df), "none", dtype=object)
        # explosive: phi >= 1
        band[np.where((~np.isnan(phi)) & (phi >= 1))[0]] = "explosive"
        # near random walk: phi < 0 OR hl > 50
        nrw_mask = (~np.isnan(phi)) & ((phi < 0) | ((~np.isnan(hl)) & (hl > 50)))
        band[np.where(nrw_mask)[0]] = "near_rw"
        # slow: 15 <= hl <= 50
        slow_mask = (~np.isnan(hl)) & (hl >= 15) & (hl <= 50)
        band[np.where(slow_mask)[0]] = "slow"
        # fast: 5 <= hl < 15
        fast_mask = (~np.isnan(hl)) & (hl >= 5) & (hl < 15)
        band[np.where(fast_mask)[0]] = "fast"
        # very_fast: hl < 5
        vf_mask = (~np.isnan(hl)) & (hl < 5) & (hl > 0)
        band[np.where(vf_mask)[0]] = "very_fast"

        for band_name in ("very_fast", "fast", "slow", "near_rw", "explosive"):
            idx = np.where(band == band_name)[0]
            if len(idx) < 30:
                continue
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                mean_ret = float(np.mean(rets)) if rets else 0.0
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"mean_reversion_halflife_w{win}_{band_name}_{side}",
                    family="mean_reversion_halflife", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=mean_ret, net_pnl=pnl,
                    params={"window": win, "band": band_name, "side": side},
                ))
    return out



# ---- new family: hurst_exponent_band (auto-implemented) ----
def mine_hurst_exponent_band(df: pd.DataFrame) -> list[PatternHit]:
    """Rolling Hurst exponent (R/S analysis). Buckets H into mean-reverting,
    random-walk, or trending bands. Distinct from instantaneous slope signs:
    captures the scaling-law character of returns over the window."""
    out: list[PatternHit] = []
    ret = df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)

    def rs_hurst(arr: np.ndarray) -> float:
        chunk_sizes = (8, 16, 32)
        log_n: list[float] = []
        log_rs: list[float] = []
        L = arr.shape[0]
        for n_chunk in chunk_sizes:
            if L < n_chunk * 2:
                continue
            rs_vals: list[float] = []
            for i in range(0, L - n_chunk + 1, n_chunk):
                chunk = arr[i:i + n_chunk]
                mean = chunk.mean()
                dev = chunk - mean
                cum = dev.cumsum()
                R = cum.max() - cum.min()
                S = chunk.std() + 1e-12
                rs_vals.append(R / S)
            if rs_vals:
                mean_rs = float(np.mean(rs_vals))
                if mean_rs > 0:
                    log_n.append(float(np.log(n_chunk)))
                    log_rs.append(float(np.log(mean_rs)))
        if len(log_n) < 2:
            return np.nan
        return float(np.polyfit(log_n, log_rs, 1)[0])

    # Buckets: mr_strong (<0.35), mr (<0.45), rw (0.45-0.55), tr (>0.55), tr_strong (>0.65)
    bucket_edges = [
        ("mr_strong", -np.inf, 0.35),
        ("mr",        0.35,    0.45),
        ("rw",        0.45,    0.55),
        ("tr",        0.55,    0.65),
        ("tr_strong", 0.65,    np.inf),
    ]

    ret_series = pd.Series(ret)
    for win in (128, 256):
        if len(ret) < win + 1:
            continue
        H = ret_series.rolling(win, min_periods=win).apply(rs_hurst, raw=True).to_numpy()
        for label, lo, hi in bucket_edges:
            mask = (H >= lo) & (H < hi) & ~np.isnan(H)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"hurst_exponent_band_w{win}_{label}_{side}",
                    family="hurst_exponent_band",
                    side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=float(np.mean(rets)) if rets else 0.0,
                    net_pnl=pnl,
                    params={"window": win, "band": label, "h_lo": lo, "h_hi": hi, "side": side},
                ))
    return out




# ---- new family: kurtosis_burst (auto-implemented) ----
def mine_kurtosis_burst(df: pd.DataFrame) -> list[PatternHit]:
    """Rolling excess kurtosis of returns to detect fat-tail clusters.
    Buckets a z-scored kurtosis series (4th-moment) so it's orthogonal to
    vol_regime's 2nd-moment signal. Each (window, band) becomes its own
    pattern, evaluated both LONG and SHORT.
    """
    out: list[PatternHit] = []
    ret = df["close"].pct_change()
    bands = [
        ("gaussian",    -np.inf, -0.5),
        ("mild_fat",    -0.5,     0.5),
        ("fat",          0.5,     1.5),
        ("extreme_fat",  1.5,    np.inf),
    ]
    for win in (50, 100, 200):
        m = ret.rolling(win, min_periods=win).mean()
        s = ret.rolling(win, min_periods=win).std() + 1e-12
        z = (ret - m) / s
        kurt = (z ** 4).rolling(win, min_periods=win).mean() - 3.0
        k_mean = kurt.rolling(win * 2, min_periods=win * 2).mean()
        k_std = kurt.rolling(win * 2, min_periods=win * 2).std() + 1e-12
        k_z = ((kurt - k_mean) / k_std).to_numpy()
        for band_name, lo, hi in bands:
            with np.errstate(invalid="ignore"):
                mask = (~np.isnan(k_z)) & (k_z >= lo) & (k_z < hi)
            idx = np.where(mask)[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                mean_ret = float(np.mean(rets)) if rets else 0.0
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"kurtosis_burst_w{win}_{band_name}_{side}",
                    family="kurtosis_burst", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=mean_ret, net_pnl=pnl,
                    params={"window": win, "band": band_name,
                            "k_z_lo": None if lo == -np.inf else lo,
                            "k_z_hi": None if hi == np.inf else hi,
                            "side": side},
                ))
    return out



# ---- new family: skew_burst (auto-implemented) ----
def mine_skew_burst(df: pd.DataFrame) -> list[PatternHit]:
    """Rolling skewness of returns banded into 5 regimes (strong_neg -> strong_pos).

    Captures distributional asymmetry over a window via the third moment of
    standardized returns. Distinct from single-bar candle asymmetry: this is
    about the *distribution* of recent returns, not the shape of any one bar.
    Strong negative skew historically precedes crash-prone regimes, strong
    positive skew often appears in squeeze conditions.
    """
    out: list[PatternHit] = []
    ret = df["close"].pct_change()
    # band edges: (-inf, -1), [-1, -0.3), [-0.3, 0.3), [0.3, 1), [1, inf)
    band_edges = [-1.0, -0.3, 0.3, 1.0]
    band_labels = ["strong_neg", "neg", "sym", "pos", "strong_pos"]
    for win in (40, 80, 160):
        m = ret.rolling(win, min_periods=win).mean()
        s = ret.rolling(win, min_periods=win).std(ddof=0) + 1e-12
        z = (ret - m) / s
        skew = (z ** 3).rolling(win, min_periods=win).mean().to_numpy()
        band = np.full_like(skew, -1, dtype=int)
        valid = ~np.isnan(skew)
        for i in range(len(skew)):
            if not valid[i]:
                continue
            v = skew[i]
            b = 0
            for e in band_edges:
                if v >= e:
                    b += 1
            band[i] = b
        for b, label in enumerate(band_labels):
            mask = (band == b)
            idx = np.where(mask)[0]
            for side in ("LONG", "SHORT"):
                wins, n, rets = _classify_outcomes(df, idx, side)
                if n < 30:
                    continue
                wlb, wub = _wilson(wins, n)
                mean_ret = float(np.mean(rets)) if rets else 0.0
                pnl = sum(rets) / 10_000 * 100
                out.append(PatternHit(
                    name=f"skew_burst_w{win}_{label}_{side}",
                    family="skew_burst", side=side,
                    n=n, wins=wins, win_rate=wins / n,
                    wilson_lower=wlb, wilson_upper=wub,
                    mean_return_bps=mean_ret, net_pnl=pnl,
                    params={"window": win, "band": label, "band_idx": b},
                ))
    return out



# ---- new family: runlength_entropy (auto-implemented) ----

# ---- family 13: run-length entropy ----------------------------------------
def mine_runlength_entropy(df: pd.DataFrame) -> list[PatternHit]:
    """Shannon entropy of the up/down run-length distribution over a rolling
    window. Distinct from `run_length` (which keys on the CURRENT streak
    count): this captures how repetitive vs diverse the sign sequence has
    been. Low entropy means a few run-lengths dominate (structured / regime);
    high entropy means runs are diverse (choppy / random)."""
    out: list[PatternHit] = []
    diff = df["close"].diff()
    sign = np.sign(diff).replace(0, np.nan).ffill().fillna(0).to_numpy()

    def _runs_entropy(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        runs: list[int] = []
        cur = arr[0]
        cnt = 1
        for v in arr[1:]:
            if v == cur:
                cnt += 1
            else:
                runs.append(cnt)
                cur = v
                cnt = 1
        runs.append(cnt)
        if not runs:
            return np.nan
        _, counts = np.unique(np.asarray(runs), return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))

    sign_ser = pd.Series(sign)
    for win in (64, 128, 256):
        H = sign_ser.rolling(win, min_periods=win).apply(_runs_entropy, raw=True).to_numpy()
        # bucket entropy into low / mid / high
        band = np.full(len(H), "", dtype=object)
        band[np.where(H < 1.0)[0]] = "low"
        band[np.where((H >= 1.0) & (H <= 2.0))[0]] = "mid"
        band[np.where(H > 2.0)[0]] = "high"
        for b_label in ("low", "mid", "high"):
            for cur_sign in (1, -1):
                mask = (band == b_label) & (sign == cur_sign) & ~np.isnan(H)
                idx = np.where(mask)[0]
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    mean_ret = float(np.mean(rets)) if rets else 0.0
                    pnl = sum(rets) / 10_000 * 100
                    sgn_lbl = "U" if cur_sign == 1 else "D"
                    out.append(PatternHit(
                        name=f"runlength_entropy_w{win}_{b_label}_{sgn_lbl}_{side}",
                        family="runlength_entropy", side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=mean_ret, net_pnl=pnl,
                        params={"window": win, "entropy_band": b_label,
                                "cur_sign": int(cur_sign), "side": side},
                    ))
    return out




# ---- new family: gap_behavior (auto-implemented) ----
def mine_gap_behavior(df: pd.DataFrame) -> list[PatternHit]:
    """Classify bar-to-bar gaps (open vs prior close, normalized by ATR) by
    magnitude/direction, crossed with prior-bar direction and intra-bar
    gap-fill state. Captures gap-fill vs gap-continuation regimes."""
    out: list[PatternHit] = []
    prev_close = df["close"].shift(1)
    prev_dir = np.sign(df["close"].shift(1) - df["close"].shift(2)).to_numpy()
    bar_range = (df["high"] - df["low"])
    atr = bar_range.rolling(14, min_periods=14).mean() + 1e-12
    gap = ((df["open"] - prev_close) / atr).to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    prev_close_arr = prev_close.to_numpy()
    gap_filled = (
        ((gap > 0) & (low <= prev_close_arr))
        | ((gap < 0) & (high >= prev_close_arr))
    )

    gap_bands = [
        ("big_down", -np.inf, -0.5),
        ("down", -0.5, -0.1),
        ("flat", -0.1, 0.1),
        ("up", 0.1, 0.5),
        ("big_up", 0.5, np.inf),
    ]
    for band_label, lo, hi in gap_bands:
        gap_mask = (gap >= lo) & (gap < hi) & ~np.isnan(gap)
        for pdir in (-1, 0, 1):
            pdir_mask = (prev_dir == pdir) & ~np.isnan(prev_dir)
            for fill_state in (True, False):
                fill_mask = (gap_filled == fill_state)
                mask = gap_mask & pdir_mask & fill_mask
                idx = np.where(mask)[0]
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    mean_ret = float(np.mean(rets)) if rets else 0.0
                    pnl = sum(rets) / 10_000 * 100
                    fstr = "filled" if fill_state else "unfilled"
                    out.append(PatternHit(
                        name=f"gap_behavior_{band_label}_pd{pdir}_{fstr}_{side}",
                        family="gap_behavior", side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=mean_ret, net_pnl=pnl,
                        params={
                            "gap_band": band_label,
                            "gap_lo": float(lo) if np.isfinite(lo) else None,
                            "gap_hi": float(hi) if np.isfinite(hi) else None,
                            "prev_dir": int(pdir),
                            "gap_filled": bool(fill_state),
                        },
                    ))
    return out



# ---- new family: price_acceleration (auto-implemented) ----
def mine_price_acceleration(df: pd.DataFrame) -> list[PatternHit]:
    """Second-derivative regime detection: cross current slope sign with
    z-scored acceleration band. Orthogonal to mtf_slope which uses only
    first-derivative signs."""
    out: list[PatternHit] = []
    close = df["close"].to_numpy()
    n_bars = len(close)

    def _lin_slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        # polyfit slope = cov(x,y)/var(x); compute directly for speed
        xm = x.mean()
        ym = arr.mean()
        denom = ((x - xm) ** 2).sum()
        if denom <= 0:
            return 0.0
        return float(((x - xm) * (arr - ym)).sum() / denom)

    accel_bands = [
        ("strong_decel", -np.inf, -1.0),
        ("decel",        -1.0,    -0.3),
        ("flat",         -0.3,     0.3),
        ("accel",         0.3,     1.0),
        ("strong_accel",  1.0,     np.inf),
    ]

    for slope_w in (5, 10, 20):
        if n_bars < slope_w * 5 + 2:
            continue
        slope_series = df["close"].rolling(slope_w).apply(_lin_slope, raw=True)
        slope_norm = (slope_series / df["close"]).to_numpy()
        accel = pd.Series(slope_norm).diff(slope_w)
        win = slope_w * 4
        a_mean = accel.rolling(win).mean()
        a_std = accel.rolling(win).std() + 1e-12
        a_z = ((accel - a_mean) / a_std).to_numpy()

        slope_sign = np.sign(slope_norm)
        valid = ~np.isnan(slope_norm) & ~np.isnan(a_z)

        for sign_val in (-1, 1):
            for band_name, lo, hi in accel_bands:
                mask = valid & (slope_sign == sign_val) & (a_z >= lo) & (a_z < hi)
                idx = np.where(mask)[0]
                if len(idx) < 30:
                    continue
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    mean_ret = float(np.mean(rets)) if rets else 0.0
                    pnl = sum(rets) / 10_000 * 100
                    sign_tag = "up" if sign_val == 1 else "dn"
                    out.append(PatternHit(
                        name=f"price_acceleration_w{slope_w}_slope{sign_tag}_{band_name}_{side}",
                        family="price_acceleration", side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=mean_ret, net_pnl=pnl,
                        params={
                            "slope_w": slope_w,
                            "slope_sign": int(sign_val),
                            "accel_band": band_name,
                            "side": side,
                        },
                    ))
    return out



# ---- new family: volume_price_divergence (auto-implemented) ----
def mine_volume_price_divergence(df: pd.DataFrame) -> list[PatternHit]:
    """Rolling correlation between log-volume and (a) absolute returns and
    (b) signed returns. Captures accumulation/distribution regimes.

    Orthogonal to volume_spike (single-bar z-score) and ma_distance
    (price-only). 3 windows x 3 abs-bands x 3 signed-bands = 27 combos per
    side (skipping ones that fail the n>=30 floor)."""
    out: list[PatternHit] = []
    close = df["close"]
    volume = df["volume"]

    ret = close.pct_change()
    abs_ret = ret.abs()
    lvol = np.log(volume.astype(float) + 1.0)

    def _band(s: pd.Series) -> np.ndarray:
        # 'neg' = -1, 'neutral' = 0, 'pos' = 1; nan -> -99 (no match)
        arr = s.to_numpy()
        out_arr = np.full(arr.shape, -99, dtype=int)
        mask_valid = ~np.isnan(arr)
        out_arr[mask_valid & (arr < -0.2)] = -1
        out_arr[mask_valid & (arr >= -0.2) & (arr <= 0.2)] = 0
        out_arr[mask_valid & (arr > 0.2)] = 1
        return out_arr

    label_map = {-1: "neg", 0: "neutral", 1: "pos"}

    for win in (20, 50, 100):
        corr_abs = lvol.rolling(win, min_periods=win).corr(abs_ret)
        corr_sgn = lvol.rolling(win, min_periods=win).corr(ret)
        band_abs = _band(corr_abs)
        band_sgn = _band(corr_sgn)
        for ba in (-1, 0, 1):
            for bs in (-1, 0, 1):
                mask = (band_abs == ba) & (band_sgn == bs)
                idx = np.where(mask)[0]
                for side in ("LONG", "SHORT"):
                    wins, n, rets = _classify_outcomes(df, idx, side)
                    if n < 30:
                        continue
                    wlb, wub = _wilson(wins, n)
                    mean_ret = float(np.mean(rets)) if rets else 0.0
                    pnl = sum(rets) / 10_000 * 100
                    out.append(PatternHit(
                        name=f"volpricediv_w{win}_abs{label_map[ba]}_sgn{label_map[bs]}_{side}",
                        family="volume_price_divergence",
                        side=side,
                        n=n, wins=wins, win_rate=wins / n,
                        wilson_lower=wlb, wilson_upper=wub,
                        mean_return_bps=mean_ret, net_pnl=pnl,
                        params={
                            "window": win,
                            "corr_abs_band": label_map[ba],
                            "corr_sgn_band": label_map[bs],
                        },
                    ))
    return out



MINERS = {
    # original 4
    "run_length": mine_run_length,
    "body_size": mine_body_size,
    "time_of_day": mine_time_of_day,
    "indicator_band": mine_indicator_bands,
    # +8 expansion families
    "candlestick": mine_candlestick,
    "vol_regime": mine_vol_regime,
    "volume_spike": mine_volume_spike,
    "ma_distance": mine_ma_distance,
    "mtf_slope": mine_mtf_slope,
    "range_compression": mine_range_compression,
    "adx_trend": mine_adx,
    "swing_count": mine_swing_count,
    # +9 statistical/structural families (auto-implemented from
    # adversarially-verified design workflow)
    "variance_of_variance": mine_variance_of_variance,
    "mean_reversion_halflife": mine_mean_reversion_halflife,
    "hurst_exponent_band": mine_hurst_exponent_band,
    "kurtosis_burst": mine_kurtosis_burst,
    "skew_burst": mine_skew_burst,
    "runlength_entropy": mine_runlength_entropy,
    "gap_behavior": mine_gap_behavior,
    "price_acceleration": mine_price_acceleration,
    "volume_price_divergence": mine_volume_price_divergence,
}


def _mine_one_timeframe(df: pd.DataFrame, tf: str, all_hits: list[PatternHit],
                        ) -> None:
    """Run every miner on this dataframe. Each hit name is suffixed with @tf
    so 5m vs 1h variants of the same pattern are tracked separately."""
    for fname, fn in MINERS.items():
        try:
            hits = fn(df)
        except Exception as exc:
            log.warning("miner %s errored on %s: %s", fname, tf, exc)
            continue
        for h in hits:
            h_tagged = PatternHit(
                name=f"{h.name}@{tf}", family=f"{h.family}_{tf}",
                side=h.side, n=h.n, wins=h.wins, win_rate=h.win_rate,
                wilson_lower=h.wilson_lower, wilson_upper=h.wilson_upper,
                mean_return_bps=h.mean_return_bps, net_pnl=h.net_pnl,
                params={**h.params, "timeframe": tf},
            )
            all_hits.append(h_tagged)


def run(symbol: str | None = None, timeframe: str | None = None,
        n_primary: int = LOOKBACK_BARS, n_holdout: int = HELD_OUT_BARS) -> dict[str, Any]:
    cfg = config.load(force=True)
    symbol = symbol or cfg.symbol
    state = _load_state()
    now_ts = config.time_now_ms()
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_run_day") == today:
        log.info("pattern mining already ran today (%s); skipping", today)
        return {"skipped": True, "day": today}

    timeframes = TIMEFRAMES_TO_MINE if timeframe is None else [timeframe]
    all_primary: list[PatternHit] = []
    all_holdout_by_name: dict[str, PatternHit] = {}
    bar_counts: dict[str, int] = {}

    for tf in timeframes:
        tf_ms = config.timeframe_ms(tf)
        # scale needed bars by timeframe so we always get ~120 days
        days_target = 120
        bars_target = days_target * 86_400_000 // tf_ms
        log.info("patterns: fetching ~%d bars of %s", bars_target, tf)
        df = _fetch(symbol, tf, int(bars_target) + n_holdout)
        if df.empty or len(df) < 1000:
            log.warning("patterns: not enough %s data (%d bars)", tf, len(df))
            continue
        bar_counts[tf] = len(df)
        # split into primary (recent) and holdout (older)
        n_holdout_tf = min(n_holdout, len(df) // 3)
        df_holdout = df.iloc[:n_holdout_tf].reset_index(drop=True)
        df_primary = df.iloc[n_holdout_tf:].reset_index(drop=True)
        log.info("  %s primary=%d holdout=%d", tf, len(df_primary), len(df_holdout))
        _mine_one_timeframe(df_primary, tf, all_primary)
        holdout_hits: list[PatternHit] = []
        _mine_one_timeframe(df_holdout, tf, holdout_hits)
        for h in holdout_hits:
            all_holdout_by_name[h.name] = h

    if not all_primary:
        log.warning("patterns: no patterns generated across any timeframe")
        return {"error": "no_data", "bar_counts": bar_counts}

    promoted_now: list[str] = []
    for hit in all_primary:
        holdout = all_holdout_by_name.get(hit.name)
        holds_out = holdout is not None and _passes_bar(holdout)
        if _maybe_promote(state, hit, holds_out, now_ts):
            promoted_now.append(hit.name)

    state["last_run_day"] = today
    state["last_run_ts"] = now_ts
    state["bar_counts"] = bar_counts
    _save_state(state)
    n_activated = _activate_promoted(state)

    passed = sum(1 for h in all_primary if _passes_bar(h))
    trial = sum(1 for h in all_primary if h.n >= TRIAL_MIN_N and h.wilson_lower > TRIAL_WLB and h.net_pnl > 0)
    summary = {
        "day": today,
        "patterns_tested": len(all_primary),
        "passed_bar_primary": passed,
        "trial_tier": trial,
        "promoted_this_run": promoted_now,
        "newly_active_in_settings": n_activated,
        "total_active_patterns": sum(
            1 for p in state["patterns"].values() if p.get("tier") == "active"
        ),
        "bar_counts": bar_counts,
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
