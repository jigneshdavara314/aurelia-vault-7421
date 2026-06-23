from __future__ import annotations

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal, edge_after_cost, kelly_size
from .. import predictions
from .base import Strategy


class ClaudePred(Strategy):
    name = "claude_pred"
    required_indicators = {
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
        "z_20": {"fn": "zscore", "args": {"n": 20}},
    }
    HORIZON_BARS = 12
    SL_ATR = 1.0
    TP_ATR = 1.5

    def evaluate(self, snap, cfg, cost_bps):
        pred = predictions.get_prediction(snap.id)
        if pred is None:
            return None
        try:
            p = float(pred["pred_p_up"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 < p < 1):
            return None
        atr_val = snap.indicators.get("atr_14")
        if atr_val is None or atr_val <= 0:
            return None
        side = "LONG" if p >= 0.5 else "SHORT"
        entry = snap.close
        if side == "LONG":
            sl = entry - self.SL_ATR * atr_val
            tp = entry + self.TP_ATR * atr_val
        else:
            sl = entry + self.SL_ATR * atr_val
            tp = entry - self.TP_ATR * atr_val
        if sl <= 0 or tp <= 0:
            return None
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            return None
        b = reward / risk
        p_eff = p if side == "LONG" else 1 - p
        edge = edge_after_cost(p_eff, b, cost_bps)
        if edge < cfg.min_edge:
            return None
        if p_eff < cfg.min_confidence:
            return None
        size = kelly_size(p_eff, b, cfg)
        if size <= 0:
            return None
        return Signal(
            snapshot=snap, strategy=self.name, side=side,
            entry_price=entry, pred_p_up=p, edge=edge, size_usd=size,
            tp_price=tp, sl_price=sl, horizon_bars=self.HORIZON_BARS,
            reason=f"claude p_up={p:.2f} {pred.get('rationale','')[:60]}",
            estimator=pred.get("estimator", "claude"),
        )
