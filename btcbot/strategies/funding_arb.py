from __future__ import annotations

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal
from .base import Strategy


class FundingArb(Strategy):
    """Scaffold: funding-rate arbitrage requires perp+spot two-leg execution.

    Phase 8 will implement the two-leg simulator. Until then this strategy
    never emits a signal and is registered only so the tournament config can
    reference it.
    """
    name = "funding_arb"
    required_indicators: dict = {}

    def evaluate(self, snap: Snapshot, cfg: StrategyConfig, cost_bps: int) -> Signal | None:
        return None
