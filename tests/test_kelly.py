from __future__ import annotations

from btcbot import config
from btcbot.strategy import edge_after_cost, kelly_size


def test_kelly_zero_for_unprofitable():
    cfg = config.profile("moderate")
    assert kelly_size(0.4, 1.0, cfg) == 0.0
    assert kelly_size(0.5, 1.0, cfg) == 0.0


def test_kelly_caps_at_max_position():
    cfg = config.profile("moderate")
    stake = kelly_size(0.95, 5.0, cfg)
    assert stake <= cfg.max_position_usd


def test_edge_after_cost_signs():
    assert edge_after_cost(0.5, 1.0, cost_bps=0) == 0.0
    assert edge_after_cost(0.6, 1.0, cost_bps=0) > 0
    assert edge_after_cost(0.51, 1.0, cost_bps=300) < 0


def test_break_even_function():
    from btcbot.backtest import break_even_win_rate
    assert abs(break_even_win_rate(1.0, 0) - 0.5) < 1e-9
    assert abs(break_even_win_rate(2.0, 0) - (1 / 3)) < 1e-9
    be = break_even_win_rate(1.0, 100)
    assert 0.5 < be < 0.55
