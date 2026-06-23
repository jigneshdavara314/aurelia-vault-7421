from __future__ import annotations

import numpy as np
import pandas as pd

from btcbot import indicators


def _make_df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    vol = rng.uniform(1, 10, n)
    ot = np.arange(n) * 300_000
    return pd.DataFrame({
        "open_time": ot, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def test_indicators_no_future_leak():
    df = _make_df(500)
    spec = {
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "rsi_14": {"fn": "rsi", "args": {"n": 14}},
        "z_20": {"fn": "zscore", "args": {"n": 20}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
    }
    full = indicators.add_indicators(df, spec)
    for cut in (120, 200, 350):
        prefix = indicators.add_indicators(df.iloc[:cut], spec)
        for col in ["ema_50", "rsi_14", "z_20", "atr_14"]:
            a = prefix[col].iloc[-1]
            b = full[col].iloc[cut - 1]
            if pd.isna(a) and pd.isna(b):
                continue
            assert np.isclose(a, b, equal_nan=False, atol=1e-9), (col, cut, a, b)


def test_rsi_in_range():
    df = _make_df(300)
    out = indicators.add_indicators(df, {"rsi_14": {"fn": "rsi", "args": {"n": 14}}})
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr_nonneg():
    df = _make_df(300)
    out = indicators.add_indicators(df, {"atr_14": {"fn": "atr", "args": {"n": 14}}})
    valid = out["atr_14"].dropna()
    assert (valid >= 0).all()


def test_regime_labels_known():
    df = _make_df(800)
    spec = {
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
    }
    out = indicators.add_indicators(df, spec)
    out = indicators.classify_regime(out)
    labels = set(out["regime"].dropna().unique().tolist())
    assert labels.issubset(indicators.valid_regimes())
