from __future__ import annotations

import pytest

from btcbot import bankroll, config, executor, store
from btcbot.data import Snapshot
from btcbot.errors import LiveDisabledError
from btcbot.strategy import Signal


def _setup():
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    bankroll.init_bankroll(strategy="nsigma_fade", mode="PAPER")


def _sig():
    snap = Snapshot(
        symbol="BTC/USDT", timeframe="5m", ts=1_000_000,
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        indicators={"atr_14": 1.0}, regime="ranging",
    )
    return Signal(
        snapshot=snap, strategy="nsigma_fade", side="LONG",
        entry_price=100.0, pred_p_up=0.55, edge=0.05, size_usd=25.0,
        tp_price=101.0, sl_price=99.0, horizon_bars=12,
        reason="test", estimator="rule",
    )


def test_paper_execute_records_trade_and_deducts_bankroll():
    _setup()
    before = bankroll.balance(strategy="nsigma_fade")
    ex = executor.Executor(exchange=None)
    sig = _sig()
    result = ex.execute(sig)
    assert result["status"] == "OPEN"
    assert result["trade_id"] > 0
    after = bankroll.balance(strategy="nsigma_fade")
    assert before - after == pytest.approx(25.0)
    assert store.open_position_count(strategy="nsigma_fade") == 1
    assert result["fill_price"] > sig.entry_price


def test_live_mode_raises_when_disabled(monkeypatch):
    _setup()
    monkeypatch.setenv("MODE", "LIVE")
    config.load(force=True)
    ex = executor.Executor(exchange=None)
    with pytest.raises(LiveDisabledError):
        ex.execute(_sig())
