from __future__ import annotations

from btcbot import bankroll, config, store


def test_init_and_status():
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    s = bankroll.summary(strategy=None, mode="PAPER")
    assert s["exists"] is True
    assert s["balance"] == config.load().initial_deposit
    assert s["open_exposure"] == 0
    assert s["drawdown_halted"] is False
    assert store.open_position_count() == 0


def test_config_loads():
    cfg = config.load(force=True)
    assert cfg.mode == "PAPER"
    assert cfg.symbol == "BTC/USDT"
    assert cfg.timeframe == "5m"
    assert "nsigma_fade" in config.active_strategies()


def test_timeframe_ms():
    assert config.timeframe_ms("5m") == 300_000
    assert config.timeframe_ms("1h") == 3_600_000
    assert config.timeframe_ms("1d") == 86_400_000
