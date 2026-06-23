from __future__ import annotations

import numpy as np
import pandas as pd

from btcbot import config
from btcbot.backtest import (
    BacktestResult,
    break_even_win_rate,
    run_backtest,
    simulate,
    wilson_interval,
)
from btcbot.strategies import get as get_strategy


def _df_drift_then_revert(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rets = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    vol = rng.uniform(1, 10, n)
    ot = np.arange(n) * 300_000
    return pd.DataFrame({
        "open_time": ot, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(50, 100)
    assert 0 < lo < 0.5 < hi < 1


def test_break_even_win_rate_increases_with_cost():
    a = break_even_win_rate(1.0, 0)
    b = break_even_win_rate(1.0, 200)
    assert b > a


def test_backtest_runs_and_returns_result():
    df = _df_drift_then_revert(800)
    res = run_backtest("nsigma_fade", df, "BTC/USDT", "5m", bankroll_usd=500.0)
    assert isinstance(res, BacktestResult)
    assert res.n_trades >= 0
    assert res.verdict in {"edge_confirmed", "no_edge", "inconclusive"}


def test_backtest_determinism_same_data():
    df = _df_drift_then_revert(500)
    a = run_backtest("nsigma_fade", df, "BTC/USDT", "5m", bankroll_usd=500.0)
    b = run_backtest("nsigma_fade", df, "BTC/USDT", "5m", bankroll_usd=500.0)
    assert a.n_trades == b.n_trades
    assert a.win_rate == b.win_rate
    assert abs(a.net_pnl - b.net_pnl) < 1e-9
