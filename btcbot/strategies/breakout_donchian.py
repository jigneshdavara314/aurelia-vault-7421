from __future__ import annotations

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal, edge_after_cost, kelly_size
from .base import Strategy


class BreakoutDonchian(Strategy):
    name = "breakout_donchian"
    required_indicators = {
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
    }
    BASE_PRED = 0.53
    HORIZON_BARS = 36
    SL_ATR = 1.5
    TP_ATR = 3.0
    BREAKOUT_BUFFER = 0.003

    def evaluate(self, snap, cfg, cost_bps):
        ind = snap.indicators
        atr_val = ind.get("atr_14")
        donch_high = ind.get("donch_high_20")
        if atr_val is None or donch_high is None or atr_val <= 0:
            return None
        if snap.regime in {"trending_down"}:
            return None
        if snap.close <= donch_high * (1 - self.BREAKOUT_BUFFER):
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
            reason=f"breakout close={entry:.2f}>donch={donch_high:.2f}",
            estimator="rule",
        )
