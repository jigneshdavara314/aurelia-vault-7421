from __future__ import annotations

from typing import Any

import numpy as np

from ..data import Snapshot
from ..strategy import Signal, edge_after_cost, kelly_size
from .base import Strategy


def _detect_run_length(snap: Snapshot, recent_closes: list[float], params: dict) -> bool:
    run_len = int(params.get("run_len", 5))
    direction = int(params.get("direction", 1))
    if len(recent_closes) < run_len + 1:
        return False
    signs = np.sign(np.diff(recent_closes[-(run_len + 1):]))
    return bool((signs == direction).all())


def _detect_body_size(snap: Snapshot, recent: list[dict], params: dict) -> bool:
    if len(recent) < 2:
        return False
    prev_class = int(params["prev_class"])
    curr_class = int(params["curr_class"])

    def _cls(bar: dict) -> int:
        body = abs(bar["close"] - bar["open"])
        rng = bar["high"] - bar["low"]
        if rng <= 0:
            return 0
        rel = body / rng
        cls = 2 if rel >= 0.6 else (1 if rel >= 0.3 else 0)
        sign = 1 if bar["close"] > bar["open"] else (-1 if bar["close"] < bar["open"] else 0)
        return cls * sign

    a, b = _cls(recent[-2]), _cls(recent[-1])
    return a == prev_class and b == curr_class


def _detect_time_of_day(snap: Snapshot, params: dict) -> bool:
    import datetime as _dt
    target_h = int(params["hour_utc"])
    bar_h = _dt.datetime.fromtimestamp(snap.ts / 1000, tz=_dt.timezone.utc).hour
    return bar_h == target_h


def _detect_indicator_band(snap: Snapshot, params: dict) -> bool:
    rsi = snap.indicators.get("rsi_14")
    z = snap.indicators.get("z_20")
    if rsi is None or z is None:
        return False
    return (params["rsi_lo"] <= rsi < params["rsi_hi"]
            and params["z_lo"] <= z < params["z_hi"])


# Module-level cache so we don't reload patterns.json every snapshot
_RECENT_CACHE: dict[str, list[dict]] = {}


def _update_recent_cache(snap: Snapshot, max_keep: int = 20) -> None:
    key = f"{snap.symbol}|{snap.timeframe}"
    buf = _RECENT_CACHE.setdefault(key, [])
    if buf and buf[-1].get("ts") == snap.ts:
        return
    buf.append({"ts": snap.ts, "open": snap.open, "high": snap.high,
                "low": snap.low, "close": snap.close})
    if len(buf) > max_keep:
        del buf[: len(buf) - max_keep]


def _recent_closes(snap: Snapshot, n: int) -> list[float]:
    key = f"{snap.symbol}|{snap.timeframe}"
    buf = _RECENT_CACHE.get(key, [])
    return [b["close"] for b in buf[-n:]]


def _recent_bars(snap: Snapshot, n: int) -> list[dict]:
    key = f"{snap.symbol}|{snap.timeframe}"
    buf = _RECENT_CACHE.get(key, [])
    return buf[-n:]


class PatternDriven(Strategy):
    """A strategy that loads its rule from patterns.json by name.

    Instantiated as PatternDriven(pattern_name='hour_13_LONG'), addressed in
    settings.json as 'pattern::hour_13_LONG'. The detection logic dispatches
    on the family stored in patterns.json.
    """
    required_indicators = {
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "rsi_14": {"fn": "rsi", "args": {"n": 14}},
        "z_20": {"fn": "zscore", "args": {"n": 20}},
    }
    HORIZON_BARS = 12
    SL_ATR = 1.0
    TP_ATR = 1.5

    def __init__(self, pattern_name: str | None = None) -> None:
        self.pattern_name = pattern_name
        self.name = f"pattern::{pattern_name}" if pattern_name else "pattern_driven"
        self._record: dict[str, Any] | None = None

    def _load_record(self) -> dict[str, Any] | None:
        if self._record is None and self.pattern_name:
            from .. import patterns as _patterns
            self._record = _patterns.get_active_pattern(self.pattern_name)
        return self._record

    def evaluate(self, snap: Snapshot, cfg, cost_bps: int) -> Signal | None:
        rec = self._load_record()
        if rec is None:
            return None
        _update_recent_cache(snap)

        family = rec["family"]
        params = rec["params"]
        side = rec["side"]
        if family == "run_length":
            if not _detect_run_length(snap, _recent_closes(snap, 10), params):
                return None
        elif family == "body_size":
            if not _detect_body_size(snap, _recent_bars(snap, 3), params):
                return None
        elif family == "time_of_day":
            if not _detect_time_of_day(snap, params):
                return None
        elif family == "indicator_band":
            if not _detect_indicator_band(snap, params):
                return None
        else:
            return None

        atr_val = snap.indicators.get("atr_14")
        if atr_val is None or atr_val <= 0:
            return None

        entry = snap.close
        if side == "LONG":
            sl = entry - self.SL_ATR * atr_val
            tp = entry + self.TP_ATR * atr_val
        else:
            sl = entry + self.SL_ATR * atr_val
            tp = entry - self.TP_ATR * atr_val
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            return None
        b = reward / risk

        # Use the mined Wilson lower bound as our calibrated p.
        # This is the *historical* win rate — the gate filters on it.
        last_hist = (rec.get("history") or [{}])[-1]
        p = max(0.51, min(0.95, last_hist.get("wlb", 0.5) + 0.02))
        edge = edge_after_cost(p, b, cost_bps)
        if edge < cfg.min_edge:
            return None
        if p < cfg.min_confidence:
            return None
        size = kelly_size(p, b, cfg)
        if size <= 0:
            return None
        return Signal(
            snapshot=snap, strategy=self.name, side=side,
            entry_price=entry, pred_p_up=p if side == "LONG" else 1 - p,
            edge=edge, size_usd=size,
            tp_price=tp, sl_price=sl, horizon_bars=self.HORIZON_BARS,
            reason=f"pattern={self.pattern_name} wlb={last_hist.get('wlb',0):.3f}",
            estimator="pattern_mined",
        )
