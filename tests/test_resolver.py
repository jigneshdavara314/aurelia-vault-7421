from __future__ import annotations

import pandas as pd

from btcbot.resolver import _resolve_one


def _candles(rows):
    return pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])


def test_resolve_tp_hit_long():
    trade = {
        "entry_ts": 1000, "side": "LONG", "tp_price": 110.0,
        "sl_price": 90.0, "timeout_ts": 1000 + 12 * 300,
    }
    c = _candles([
        (1300, 100, 105, 99, 102, 1),
        (1600, 102, 112, 101, 111, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out["reason"] == "TP"
    assert out["price"] == 110.0


def test_resolve_sl_hit_long():
    trade = {
        "entry_ts": 1000, "side": "LONG", "tp_price": 110.0,
        "sl_price": 90.0, "timeout_ts": 1000 + 12 * 300,
    }
    c = _candles([
        (1300, 100, 102, 89, 95, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out["reason"] == "SL"
    assert out["price"] == 90.0


def test_resolve_sl_before_tp_when_both_in_same_bar():
    trade = {
        "entry_ts": 1000, "side": "LONG", "tp_price": 110.0,
        "sl_price": 90.0, "timeout_ts": 1000 + 12 * 300,
    }
    c = _candles([
        (1300, 100, 112, 89, 100, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out["reason"] == "SL"


def test_resolve_timeout():
    trade = {
        "entry_ts": 1000, "side": "LONG", "tp_price": 110.0,
        "sl_price": 90.0, "timeout_ts": 1900,
    }
    c = _candles([
        (1300, 100, 105, 95, 102, 1),
        (1600, 102, 106, 100, 103, 1),
        (1900, 103, 107, 100, 104, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out["reason"] == "TIMEOUT"


def test_resolve_short_tp():
    trade = {
        "entry_ts": 1000, "side": "SHORT", "tp_price": 90.0,
        "sl_price": 110.0, "timeout_ts": 1900,
    }
    c = _candles([
        (1300, 100, 102, 88, 89, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out["reason"] == "TP"
    assert out["price"] == 90.0


def test_resolve_not_yet():
    trade = {
        "entry_ts": 1000, "side": "LONG", "tp_price": 110.0,
        "sl_price": 90.0, "timeout_ts": 9999,
    }
    c = _candles([
        (1300, 100, 105, 99, 102, 1),
    ])
    out = _resolve_one(trade, c, 300)
    assert out is None
