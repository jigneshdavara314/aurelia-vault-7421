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
    Z_THRESH = 0.7
    SL_ATR = 1.5
    TP_ATR = 2.5
    HORIZON_BARS = 24
    BASE_PRED = 0.53

    def evaluate(self, snap, cfg, cost_bps):
        ind = snap.indicators
        if "z_20" not in ind or "atr_14" not in ind:
            return None
        z = ind["z_20"]
        atr_val = ind["atr_14"]
        if atr_val <= 0:
            return None
        if snap.regime in {"trending_up"}:
            side = "SHORT" if z >= self.Z_THRESH else None
        elif snap.regime in {"trending_down"}:
            side = "LONG" if z <= -self.Z_THRESH else None
        else:
            if z <= -self.Z_THRESH:
                side = "LONG"
            elif z >= self.Z_THRESH:
                side = "SHORT"
            else:
                side = None
        if side is None:
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
            snapshot=snap, strategy=self.name, side=side,
            entry_price=entry, pred_p_up=p if side == "LONG" else 1 - p,
            edge=edge, size_usd=size,
            tp_price=tp, sl_price=sl, horizon_bars=self.HORIZON_BARS,
            reason=f"z={z:.2f} regime={snap.regime}",
            estimator="rule",
        )
