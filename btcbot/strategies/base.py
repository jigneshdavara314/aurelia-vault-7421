from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import StrategyConfig
from ..data import Snapshot
from ..strategy import Signal


class Strategy(ABC):
    name: str = "abstract"
    required_indicators: dict = {}

    @abstractmethod
    def evaluate(
        self, snapshot: Snapshot, cfg: StrategyConfig, cost_bps: int,
    ) -> Signal | None: ...
