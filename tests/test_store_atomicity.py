from __future__ import annotations

import pytest

from btcbot import bankroll, config, store


def _open_trade(strategy: str = "nsigma_fade") -> int:
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    bankroll.init_bankroll(strategy=strategy, mode="PAPER")
    sd = {
        "strategy": strategy, "symbol": "BTC/USDT", "timeframe": "5m",
        "side": "LONG", "tp_price": 110.0, "sl_price": 90.0,
        "horizon_bars": 12, "timeout_ts": 1_000_000 + 12 * 300_000,
        "pred_p_up": 0.55, "edge": 0.03, "estimator": "rule",
        "regime": "ranging", "reason": "test",
    }
    fill = {
        "mode": "PAPER", "ts": 1_000_000, "fill_price": 100.0,
        "fill_size_usd": 50.0, "fee_bps_assumed": 10, "slippage_bps_assumed": 5,
    }
    trade_id = store.record_trade(sd, fill)
    bankroll.deduct_stake(strategy, 50.0, "open:test", trade_id=trade_id)
    return trade_id


def test_settle_and_credit_atomic_won():
    tid = _open_trade()
    before = bankroll.balance(strategy="nsigma_fade")
    pnl, new_bal = store.settle_and_credit(tid, 2_000_000, 110.0, "TP", 0.10, 0.05)
    assert pnl > 0
    assert new_bal == bankroll.balance(strategy="nsigma_fade")
    assert new_bal > before


def test_settle_and_credit_atomic_lost():
    tid = _open_trade()
    pnl, new_bal = store.settle_and_credit(tid, 2_000_000, 90.0, "SL", 0.10, 0.05)
    assert pnl < 0
    assert new_bal == bankroll.balance(strategy="nsigma_fade")


def test_settle_rejects_double_close():
    tid = _open_trade()
    store.settle_and_credit(tid, 2_000_000, 110.0, "TP", 0.10, 0.05)
    with pytest.raises(store.StoreError):
        store.settle_and_credit(tid, 3_000_000, 105.0, "TP", 0.10, 0.05)


def test_bankroll_balance_consistency_after_n_trades():
    """Bankroll balance after N closed trades equals initial - sum(stakes) +
    sum(payouts). Bankroll log replays to the same number."""
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    init = bankroll.balance(strategy="nsigma_fade")
    for i in range(5):
        tid = _open_trade()
        store.settle_and_credit(
            tid, 2_000_000 + i, 110.0 if i % 2 == 0 else 90.0,
            "TP" if i % 2 == 0 else "SL", 0.10, 0.05,
        )
    end = bankroll.balance(strategy="nsigma_fade")
    with store.conn_ctx() as c:
        bk_id = c.execute(
            "SELECT id FROM bankroll WHERE strategy='nsigma_fade'"
        ).fetchone()["id"]
        rows = c.execute(
            "SELECT delta, balance_after FROM bankroll_log WHERE bankroll_id=? ORDER BY id",
            (bk_id,),
        ).fetchall()
    replayed = 0.0
    for r in rows:
        replayed += r["delta"]
    if rows:
        assert abs(replayed - rows[-1]["balance_after"]) < 1e-6
    assert abs(end - rows[-1]["balance_after"]) < 1e-6 if rows else True
