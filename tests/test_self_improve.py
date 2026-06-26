from __future__ import annotations

import json

from btcbot import bankroll, config, self_improve, store


def _seed_trades(strategy: str, regime: str, side: str, n_won: int, n_lost: int):
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    bankroll.init_bankroll(strategy=strategy, mode="PAPER")
    base_ts = 1_000_000
    for i in range(n_won + n_lost):
        sd = {
            "strategy": strategy, "symbol": "BTC/USDT", "timeframe": "5m",
            "side": side, "tp_price": 110.0, "sl_price": 90.0,
            "horizon_bars": 12, "timeout_ts": base_ts + i * 1000 + 12 * 300_000,
            "pred_p_up": 0.55, "edge": 0.03, "estimator": "rule",
            "regime": regime, "reason": "seed",
        }
        fill = {
            "mode": "PAPER", "ts": base_ts + i * 1000, "fill_price": 100.0,
            "fill_size_usd": 10.0, "fee_bps_assumed": 10, "slippage_bps_assumed": 5,
        }
        tid = store.record_trade(sd, fill)
        bankroll.deduct_stake(strategy, 10.0, "open:test", trade_id=tid)
        won = i < n_won
        store.settle_and_credit(
            tid, base_ts + i * 1000 + 1, 110.0 if won else 90.0,
            "TP" if won else "SL", 0.10, 0.05,
        )


def test_self_improve_promotes_high_winrate_cell():
    _seed_trades("nsigma_fade", "ranging", "LONG", n_won=22, n_lost=3)
    now = config.time_now_ms()
    out = self_improve.run(now)
    assert out["evaluated"] >= 1
    state = self_improve.all_states()
    # Cells are collapsed across regime now; key is strategy|*|side
    key = "nsigma_fade|*|LONG"
    assert key in state
    assert state[key].rolling_wilson_lower > 0.5


def test_self_improve_demotes_terrible_cell():
    _seed_trades("nsigma_fade", "ranging", "LONG", n_won=2, n_lost=30)
    now = config.time_now_ms()
    self_improve.run(now)
    state = self_improve.all_states()
    # Cells are collapsed across regime now; key is strategy|*|side
    key = "nsigma_fade|*|LONG"
    assert state[key].tier in {"trial", "disabled"}


def test_strategy_state_is_atomic_written():
    _seed_trades("nsigma_fade", "ranging", "LONG", n_won=10, n_lost=10)
    self_improve.run(config.time_now_ms())
    raw = json.loads(config.STRATEGY_STATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert any(k.startswith("nsigma_fade|") for k in raw)
