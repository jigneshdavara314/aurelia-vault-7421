from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass

from . import bankroll, config, store
from .strategy import Signal


@dataclass(frozen=True)
class GateResult:
    ok: bool
    gate: str = ""
    reason: str = ""


PASS = GateResult(True)


def _day_start_ms(now_ts: int) -> int:
    d = _dt.datetime.fromtimestamp(now_ts / 1000, tz=_dt.timezone.utc).date()
    return int(_dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc).timestamp() * 1000)


def gate_active_strategy(strategy: str) -> GateResult:
    if strategy not in config.active_strategies():
        return GateResult(False, "active_strategy", f"{strategy} not in active list")
    return PASS


def gate_drawdown_halt(strategy: str | None) -> GateResult:
    if bankroll.drawdown_halted(strategy):
        return GateResult(False, "drawdown_halt", "drawdown halt active")
    return PASS


def gate_min_edge(signal: Signal, scfg) -> GateResult:
    if signal.edge < scfg.min_edge:
        return GateResult(False, "min_edge", f"edge={signal.edge:.4f} < {scfg.min_edge}")
    return PASS


def gate_min_confidence(signal: Signal, scfg) -> GateResult:
    if signal.pred_p_up is not None and signal.pred_p_up < scfg.min_confidence:
        return GateResult(False, "min_confidence",
                          f"p_up={signal.pred_p_up:.3f} < {scfg.min_confidence}")
    return PASS


def gate_already_open(symbol: str, entry_bar_ts: int, strategy: str) -> GateResult:
    if store.already_open(symbol, entry_bar_ts, strategy):
        return GateResult(False, "already_open", "duplicate")
    return PASS


def gate_position_cap(gcfg) -> GateResult:
    if store.open_position_count() >= gcfg.max_open_positions:
        return GateResult(False, "position_cap", f"{gcfg.max_open_positions}")
    return PASS


def gate_per_symbol_cap(symbol: str, gcfg) -> GateResult:
    if store.open_count_for_symbol(symbol) >= gcfg.max_open_per_symbol:
        return GateResult(False, "per_symbol_cap", f"{gcfg.max_open_per_symbol}")
    return PASS


def gate_daily_spend(strategy: str, stake: float, now_ts: int) -> GateResult:
    spent = store.staked_today(strategy, _day_start_ms(now_ts))
    budget = config.daily_budget()
    if spent + stake > budget:
        return GateResult(False, "daily_spend", f"spent={spent:.2f} budget={budget:.2f}")
    return PASS


def gate_exposure(strategy: str | None, stake: float) -> GateResult:
    if not bankroll.exposure_ok(strategy, stake):
        return GateResult(False, "exposure", "aggregate exposure ceiling")
    return PASS


def gate_fillable_depth(signal: Signal, exchange) -> GateResult:
    try:
        side = "buy" if signal.side == "LONG" else "sell"
        depth = exchange.fillable_depth(signal.snapshot.symbol, side, max_slippage_bps=15)
    except Exception as exc:
        return GateResult(False, "fillable_depth", f"depth fetch failed: {exc}")
    if depth < signal.size_usd:
        return GateResult(False, "fillable_depth", f"depth={depth:.0f} < size={signal.size_usd}")
    return PASS


def gate_affordable(strategy: str | None, stake: float) -> GateResult:
    if not bankroll.can_afford(strategy, stake):
        return GateResult(False, "affordable", f"insufficient bankroll for {stake}")
    return PASS


def run_gates(signal: Signal, exchange, now_ts: int) -> GateResult:
    gcfg = config.load()
    scfg = config.profile(gcfg.profile)
    for g in (
        gate_active_strategy(signal.strategy),
        gate_drawdown_halt(signal.strategy),
        gate_min_edge(signal, scfg),
        gate_min_confidence(signal, scfg),
        gate_already_open(signal.snapshot.symbol, signal.snapshot.ts, signal.strategy),
        gate_position_cap(gcfg),
        gate_per_symbol_cap(signal.snapshot.symbol, gcfg),
        gate_daily_spend(signal.strategy, signal.size_usd, now_ts),
        gate_exposure(signal.strategy, signal.size_usd),
    ):
        if not g.ok:
            store.record_gate_failure(
                now_ts, signal.strategy, signal.snapshot.symbol,
                g.gate, g.reason, size_usd=signal.size_usd,
                pred_p_up=signal.pred_p_up,
            )
            return g
    if exchange is not None:
        d = gate_fillable_depth(signal, exchange)
        if not d.ok:
            store.record_gate_failure(
                now_ts, signal.strategy, signal.snapshot.symbol,
                d.gate, d.reason, size_usd=signal.size_usd,
                pred_p_up=signal.pred_p_up,
            )
            return d
    a = gate_affordable(signal.strategy, signal.size_usd)
    if not a.ok:
        store.record_gate_failure(
            now_ts, signal.strategy, signal.snapshot.symbol,
            a.gate, a.reason, size_usd=signal.size_usd,
            pred_p_up=signal.pred_p_up,
        )
        return a
    return PASS


def should_attempt(signal: Signal, mode: str, now_ts: int, paper_fill_prob: float = 0.95) -> bool:
    if mode == "LIVE":
        return True
    day = now_ts // 86_400_000
    h = hashlib.sha256(f"{signal.snapshot.id}|{day}".encode()).digest()
    draw = int.from_bytes(h[:4], "big") / 2**32
    return draw < paper_fill_prob
