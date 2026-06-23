from __future__ import annotations

import pytest

from btcbot import bankroll, config, engine, store
from btcbot.data import Snapshot
from btcbot.strategy import Signal


def _snap(ts=1_000_000) -> Snapshot:
    return Snapshot(
        symbol="BTC/USDT", timeframe="5m", ts=ts,
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        indicators={"atr_14": 1.0}, regime="ranging",
    )


def _sig(strategy="nsigma_fade", size=10.0, p=0.55, edge=0.05) -> Signal:
    s = _snap()
    return Signal(
        snapshot=s, strategy=strategy, side="LONG", entry_price=100.0,
        pred_p_up=p, edge=edge, size_usd=size, tp_price=101.0, sl_price=99.0,
        horizon_bars=12, reason="test", estimator="rule",
    )


def _setup():
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    bankroll.init_bankroll(strategy="nsigma_fade", mode="PAPER")


def test_gate_active_strategy_unknown_blocks():
    _setup()
    g = engine.gate_active_strategy("not_in_settings")
    assert not g.ok


def test_gate_min_edge_blocks_low_edge():
    _setup()
    sig = _sig(edge=-0.01)
    scfg = config.profile("moderate")
    assert not engine.gate_min_edge(sig, scfg).ok


def test_gate_drawdown_halt_off_by_default():
    _setup()
    assert engine.gate_drawdown_halt("nsigma_fade").ok


def test_gate_drawdown_halt_triggers_when_set():
    _setup()
    bankroll.manual_set_halt(strategy="nsigma_fade", halted=True)
    assert not engine.gate_drawdown_halt("nsigma_fade").ok


def test_gate_daily_spend_blocks_over_budget():
    _setup()
    now = config.time_now_ms()
    g = engine.gate_daily_spend("nsigma_fade", stake=999_999.0, now_ts=now)
    assert not g.ok


def test_gate_position_cap():
    _setup()
    gcfg = config.load()
    g = engine.gate_position_cap(gcfg)
    assert g.ok


def test_run_gates_full_pass():
    _setup()
    sig = _sig()
    now = config.time_now_ms()
    g = engine.run_gates(sig, exchange=None, now_ts=now)
    assert g.ok, (g.gate, g.reason)


def test_run_gates_blocks_when_strategy_inactive(monkeypatch):
    _setup()
    monkeypatch.setattr(config, "active_strategies", lambda: [])
    sig = _sig()
    g = engine.run_gates(sig, exchange=None, now_ts=config.time_now_ms())
    assert not g.ok
    assert g.gate == "active_strategy"


def test_should_attempt_is_deterministic_within_day():
    _setup()
    sig = _sig()
    now = config.time_now_ms()
    a = engine.should_attempt(sig, "PAPER", now)
    b = engine.should_attempt(sig, "PAPER", now + 1000)
    assert a == b
