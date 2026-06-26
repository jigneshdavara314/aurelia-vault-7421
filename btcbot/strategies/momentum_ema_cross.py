from __future__ import annotations

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal, edge_after_cost, kelly_size
from .base import Strategy


class MomentumEmaCross(Strategy):
    name = "momentum_ema_cross"
    required_indicators = {
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
    }
    BASE_PRED = 0.53
    HORIZON_BARS = 36
    SL_ATR = 1.5
    TP_ATR = 3.5
    PULLBACK_ATR = 3.0

    def evaluate(self, snap, cfg, cost_bps):
        ind = snap.indicators
        atr_val = ind.get("atr_14")
        e50 = ind.get("ema_50")
        e200 = ind.get("ema_200")
        if atr_val is None or e50 is None or e200 is None or atr_val <= 0:
            return None
        if e50 <= e200:
            return None
        if snap.regime in {"trending_down"}:
            return None
        if abs(snap.close - e50) > self.PULLBACK_ATR * atr_val:
            return None
        entry = snap.close
        sl = entry - self.SL_ATR * atr_val
        tp = entry + self.TP_ATR * atr_val
        if sl <= 0 or tp <= entry:
            return None
        b = (tp - entry) / (entry - sl)
        p = self.BASE_PRED
        edge = edge_after_cost(p, b, cost_bps)
        if edge < cfg.min_edge:
            return None
        if p < cfg.min_confidence:
            return None
        size = kelly_size(p, b, cfg)
        if size <= 0:
            return None
        return Signal(
            snapshot=snap, strategy=self.name, side="LONG",
            entry_price=entry, pred_p_up=p, edge=edge, size_usd=size,
            tp_price=tp, sl_price=sl, horizon_bars=self.HORIZON_BARS,
            reason=f"momo e50={e50:.2f}>e200={e200:.2f}",
            estimator="rule",
        )
