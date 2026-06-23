from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import StrategyConfig
from .data import Snapshot


@dataclass(frozen=True)
class Signal:
    snapshot: Snapshot
    strategy: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    pred_p_up: float | None
    edge: float
    size_usd: float
    tp_price: float
    sl_price: float
    horizon_bars: int
    reason: str
    estimator: str

    def to_dict(self, timeout_ts: int) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.snapshot.symbol,
            "timeframe": self.snapshot.timeframe,
            "side": self.side,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "horizon_bars": self.horizon_bars,
            "timeout_ts": timeout_ts,
            "pred_p_up": self.pred_p_up,
            "edge": self.edge,
            "estimator": self.estimator,
            "regime": self.snapshot.regime,
            "reason": self.reason,
        }


def kelly_size(p: float, b: float, cfg: StrategyConfig) -> float:
    if b <= 0 or not (0 < p < 1):
        return 0.0
    q = 1 - p
    f = (b * p - q) / b
    f = max(0.0, f) * cfg.kelly_fraction
    stake = f * cfg.bankroll_usd
    return round(min(stake, cfg.max_position_usd), 2)


def edge_after_cost(p: float, b: float, cost_bps: int) -> float:
    """Expected return on capital, per Kelly framing, net of round-trip cost."""
    if b <= 0 or not (0 < p < 1):
        return -1.0
    expected_return_per_dollar = b * p - (1 - p)
    return expected_return_per_dollar - cost_bps / 10_000
