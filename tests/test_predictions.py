from __future__ import annotations

from btcbot import config, predictions


def test_append_and_get_prediction():
    predictions.clear_cache()
    row = {
        "snapshot_id": "BTC/USDT|5m|123",
        "pred_p_up": 0.62, "estimator": "claude", "ts": 1_000_000,
    }
    predictions.append_prediction(row)
    predictions.clear_cache()
    got = predictions.get_prediction("BTC/USDT|5m|123")
    assert got is not None
    assert got["pred_p_up"] == 0.62
    assert got["estimator"] == "claude"


def test_missing_prediction_returns_none():
    predictions.clear_cache()
    assert predictions.get_prediction("missing|5m|0") is None


def test_export_writes_jsonl():
    from btcbot.data import Snapshot
    snaps = [
        Snapshot(symbol="BTC/USDT", timeframe="5m", ts=ts,
                 open=100, high=101, low=99, close=100, volume=1,
                 indicators={"atr_14": 1.0}, regime="ranging")
        for ts in range(1_000_000, 1_000_500, 100)
    ]
    path = predictions.export_snapshots_todo(snaps, prompt_template="hello")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") >= len(snaps)
