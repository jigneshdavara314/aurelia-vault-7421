from __future__ import annotations

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal, edge_after_cost, kelly_size
from .base import Strategy


class NSigmaFade(Strategy):
    name = "nsigma_fade"
    required_indicators = {
        "z_20": {"fn": "zscore", "args": {"n": 20}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
    }
    Z_THRESH = -1.5
    SL_ATR = 0.7
    TP_ATR = 1.2
    HORIZON_BARS = 12
    BASE_PRED = 0.55

    def evaluate(self, snap, cfg, cost_bps):
        ind = snap.indicators
        if "z_20" not in ind or "atr_14" not in ind:
            return None
        z = ind["z_20"]
        atr_val = ind["atr_14"]
        if atr_val <= 0:
            return None
        if snap.regime not in {"ranging", "mixed"}:
            return None
        if z >= self.Z_THRESH:
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
            reason=f"z={z:.2f} atr={atr_val:.2f}",
            estimator="rule",
        )
