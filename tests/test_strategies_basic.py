from __future__ import annotations

from btcbot import config
from btcbot.data import Snapshot
from btcbot.strategies import get


def _snap(close: float, atr: float = 1.0, z: float = -2.5, regime: str = "ranging",
          extra: dict | None = None) -> Snapshot:
    ind = {"atr_14": atr, "z_20": z, "ema_50": close, "ema_200": close * 0.99}
    if extra:
        ind.update(extra)
    return Snapshot(
        symbol="BTC/USDT", timeframe="5m", ts=1_000_000,
        open=close, high=close + 0.5, low=close - 0.5, close=close, volume=1.0,
        indicators=ind, regime=regime,
    )


def test_nsigma_fade_fires_on_negative_z_in_range():
    strat = get("nsigma_fade")
    sig = strat.evaluate(_snap(100.0), config.profile("moderate"), cost_bps=20)
    assert sig is not None
    assert sig.side == "LONG"
    assert sig.size_usd > 0


def test_nsigma_fade_silent_on_positive_z():
    strat = get("nsigma_fade")
    sig = strat.evaluate(_snap(100.0, z=0.5), config.profile("moderate"), cost_bps=20)
    assert sig is None


def test_nsigma_fade_silent_in_trending():
    strat = get("nsigma_fade")
    sig = strat.evaluate(_snap(100.0, regime="trending_up"), config.profile("moderate"), cost_bps=20)
    assert sig is None


def test_breakout_requires_donch_break():
    strat = get("breakout_donchian")
    snap = _snap(100.0, regime="trending_up", extra={"donch_high_20": 99.0})
    sig = strat.evaluate(snap, config.profile("moderate"), cost_bps=20)
    assert sig is not None


def test_breakout_silent_below_donch():
    strat = get("breakout_donchian")
    snap = _snap(100.0, regime="trending_up", extra={"donch_high_20": 105.0})
    sig = strat.evaluate(snap, config.profile("moderate"), cost_bps=20)
    assert sig is None


def test_momentum_requires_ema50_above_ema200():
    strat = get("momentum_ema_cross")
    # close just above ema_50, which is above ema_200 -> valid pullback long
    snap = _snap(100.2, regime="trending_up",
                 extra={"ema_50": 100.0, "ema_200": 99.0})
    sig = strat.evaluate(snap, config.profile("moderate"), cost_bps=20)
    assert sig is not None
    # ema_50 below ema_200 -> no signal
    sig = strat.evaluate(_snap(100.0, regime="trending_up",
                                extra={"ema_50": 95.0, "ema_200": 99.0}),
                          config.profile("moderate"), cost_bps=20)
    assert sig is None


def test_funding_arb_is_scaffold_only():
    strat = get("funding_arb")
    sig = strat.evaluate(_snap(100.0), config.profile("moderate"), cost_bps=20)
    assert sig is None


def test_claude_pred_silent_without_prediction():
    strat = get("claude_pred")
    sig = strat.evaluate(_snap(100.0), config.profile("moderate"), cost_bps=20)
    assert sig is None
