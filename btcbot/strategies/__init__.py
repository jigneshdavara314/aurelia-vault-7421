from __future__ import annotations

from .base import Strategy
from .nsigma_fade import NSigmaFade
from .breakout_donchian import BreakoutDonchian
from .momentum_ema_cross import MomentumEmaCross
from .claude_pred import ClaudePred
from .funding_arb import FundingArb

REGISTRY: dict[str, type[Strategy]] = {
    "nsigma_fade": NSigmaFade,
    "breakout_donchian": BreakoutDonchian,
    "momentum_ema_cross": MomentumEmaCross,
    "claude_pred": ClaudePred,
    "funding_arb": FundingArb,
}


def get(name: str) -> Strategy:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown strategy {name!r}")
    return cls()


def names() -> list[str]:
    return list(REGISTRY.keys())
