from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def realized_vol(s: pd.Series, n: int = 288) -> pd.Series:
    r = s.pct_change()
    return r.rolling(n, min_periods=n).std(ddof=0)


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    return (
        df["high"].rolling(n, min_periods=n).max().shift(1),
        df["low"].rolling(n, min_periods=n).min().shift(1),
    )


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def volume_zscore(df: pd.DataFrame, n: int = 288) -> pd.Series:
    v = df["volume"]
    m = v.rolling(n, min_periods=n).mean()
    sd = v.rolling(n, min_periods=n).std(ddof=0)
    return (v - m) / sd.replace(0, np.nan)


def _directional_movement(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx, plus_di, minus_di


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    a, _, _ = _directional_movement(df, n)
    return a


def plus_di(df: pd.DataFrame, n: int = 14) -> pd.Series:
    _, p, _ = _directional_movement(df, n)
    return p


def minus_di(df: pd.DataFrame, n: int = 14) -> pd.Series:
    _, _, m = _directional_movement(df, n)
    return m


_REGISTRY: dict[str, Callable] = {
    "ema": lambda df, n: ema(df["close"], n),
    "sma": lambda df, n: sma(df["close"], n),
    "rsi": lambda df, n=14: rsi(df["close"], n),
    "atr": lambda df, n=14: atr(df, n),
    "zscore": lambda df, n=20: zscore(df["close"], n),
    "realized_vol": lambda df, n=288: realized_vol(df["close"], n),
    "volume_zscore": lambda df, n=288: volume_zscore(df, n),
    "adx": lambda df, n=14: adx(df, n),
    "plus_di": lambda df, n=14: plus_di(df, n),
    "minus_di": lambda df, n=14: minus_di(df, n),
}


def add_indicators(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = df.copy()
    for name, conf in spec.items():
        fn = _REGISTRY.get(conf["fn"])
        if fn is None:
            raise ValueError(f"unknown indicator fn {conf['fn']!r}")
        args = conf.get("args", {})
        out[name] = fn(out, **args)
    return out


_VALID_REGIMES = {"trending_up", "trending_down", "ranging", "high_vol", "mixed"}


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic regime label per bar. Requires ema_50, ema_200, atr_14 columns."""
    out = df.copy()
    needed = {"ema_50", "ema_200", "atr_14"}
    missing = needed - set(out.columns)
    if missing:
        spec = {
            "ema_50": {"fn": "ema", "args": {"n": 50}},
            "ema_200": {"fn": "ema", "args": {"n": 200}},
            "atr_14": {"fn": "atr", "args": {"n": 14}},
        }
        spec = {k: v for k, v in spec.items() if k in missing}
        out = add_indicators(out, spec)

    close = out["close"]
    e50 = out["ema_50"]
    e200 = out["ema_200"]
    atr_pct = (out["atr_14"] / close).replace([np.inf, -np.inf], np.nan)

    p50 = atr_pct.rolling(2000, min_periods=200).quantile(0.5)
    p75 = atr_pct.rolling(2000, min_periods=200).quantile(0.75)
    p90 = atr_pct.rolling(2000, min_periods=200).quantile(0.9)

    slope_50 = e50.diff(20)

    regime = pd.Series("mixed", index=out.index, dtype="object")
    regime[(close > e200) & (slope_50 > 0) & (atr_pct < p75)] = "trending_up"
    regime[(close < e200) & (slope_50 < 0) & (atr_pct < p75)] = "trending_down"
    ranging_mask = ((close - e200).abs() / close < 0.02) & (atr_pct < p50)
    regime[ranging_mask] = "ranging"
    regime[atr_pct > p90] = "high_vol"
    regime[atr_pct.isna() | e200.isna() | e50.isna()] = "mixed"
    out["regime"] = regime.astype("object")
    return out


def valid_regimes() -> set[str]:
    return set(_VALID_REGIMES)
