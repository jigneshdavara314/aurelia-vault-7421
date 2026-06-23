from __future__ import annotations

import json

from .. import config
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


def _resolve_variant(name: str) -> Strategy | None:
    """Discovered variants are stored in discoveries.json as
    'parent::variant_name'. Build the subclass on-the-fly."""
    if "::" not in name:
        return None
    parent_name, _ = name.split("::", 1)
    parent_cls = REGISTRY.get(parent_name)
    if parent_cls is None:
        return None
    disc_path = config.ROOT / "discoveries.json"
    if not disc_path.exists():
        return None
    try:
        state = json.loads(disc_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    var = state.get("variants", {}).get(name)
    if var is None:
        return None

    class _V(parent_cls):
        pass
    _V.name = name
    inst = _V()
    for k, v in (var.get("params") or {}).items():
        setattr(inst, k, v)
    return inst


def get(name: str) -> Strategy:
    cls = REGISTRY.get(name)
    if cls is not None:
        return cls()
    inst = _resolve_variant(name)
    if inst is not None:
        return inst
    raise ValueError(f"unknown strategy {name!r}")


def names() -> list[str]:
    return list(REGISTRY.keys())
