from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config, indicators
from .data import Snapshot, replay
from .strategies import get as get_strategy
from .strategies.base import Strategy


@dataclass
class BacktestTrade:
    entry_ts: int
    entry_price: float
    side: str
    size_usd: float
    tp_price: float
    sl_price: float
    horizon_bars: int
    pred_p_up: float | None
    edge: float
    strategy: str
    regime: str | None
    reason: str
    estimator: str
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_usd: float | None = None
    fee_usd: float | None = None
    slippage_usd: float | None = None
    bars_held: int | None = None


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    timeframe: str
    start: int
    end: int
    n_trades: int
    n_won: int
    n_lost: int
    n_timeout: int
    win_rate: float
    win_rate_wilson_lower: float
    win_rate_wilson_upper: float
    gross_pnl: float
    fee_paid: float
    slippage_paid: float
    net_pnl: float
    sharpe: float
    max_drawdown_pct: float
    avg_holding_bars: float
    break_even_win_rate: float
    verdict: str
    trades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def break_even_win_rate(avg_b: float, cost_bps: int) -> float:
    """Win rate at which net expected return is 0 given payoff ratio b and round-trip cost."""
    if avg_b <= 0:
        return 1.0
    cost = cost_bps / 10_000
    # b*p - (1-p) - cost = 0  -> p = (1 + cost) / (1 + b)
    return min(1.0, max(0.0, (1.0 + cost) / (1.0 + avg_b)))


def _resolve_one_in_replay(
    trade: BacktestTrade, future: list[tuple[int, float, float, float, float]],
    timeframe_ms: int, fee_bps: int, slippage_bps: int,
) -> BacktestTrade:
    """future is list of (ts, open, high, low, close) bars STRICTLY after entry bar."""
    timeout_ts = trade.entry_ts + trade.horizon_bars * timeframe_ms
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    bars_held = 0
    for ts, _o, h, l, c in future:
        bars_held += 1
        if ts > timeout_ts:
            exit_ts = ts
            exit_price = c
            exit_reason = "TIMEOUT"
            break
        if trade.side == "LONG":
            sl_hit = l <= trade.sl_price
            tp_hit = h >= trade.tp_price
        else:
            sl_hit = h >= trade.sl_price
            tp_hit = l <= trade.tp_price
        if sl_hit and tp_hit:
            exit_ts = ts
            exit_price = trade.sl_price
            exit_reason = "SL"
            break
        if sl_hit:
            exit_ts = ts
            exit_price = trade.sl_price
            exit_reason = "SL"
            break
        if tp_hit:
            exit_ts = ts
            exit_price = trade.tp_price
            exit_reason = "TP"
            break
    if exit_ts is None:
        if not future:
            return trade
        ts, _o, _h, _l, c = future[-1]
        exit_ts = ts
        exit_price = c
        exit_reason = "TIMEOUT"
        bars_held = len(future)

    qty = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0.0
    if trade.side == "LONG":
        gross = qty * (exit_price - trade.entry_price)
    else:
        gross = qty * (trade.entry_price - exit_price)
    fee_usd = trade.size_usd * (fee_bps / 10_000) * 2
    slip_usd = trade.size_usd * (slippage_bps / 10_000) * 2
    pnl = gross - fee_usd - slip_usd

    trade.exit_ts = exit_ts
    trade.exit_price = float(exit_price)
    trade.exit_reason = exit_reason
    trade.bars_held = bars_held
    trade.fee_usd = fee_usd
    trade.slippage_usd = slip_usd
    trade.pnl_usd = pnl
    return trade


def simulate(
    strategy: Strategy, df: pd.DataFrame,
    symbol: str, timeframe: str,
    cfg_strategy, gcfg, bankroll_usd: float,
    cost_bps: int | None = None,
) -> BacktestResult:
    if cost_bps is None:
        cost_bps = gcfg.paper_fee_bps + gcfg.paper_slippage_bps + 5
    tf_ms = config.timeframe_ms(timeframe)
    spec = dict(strategy.required_indicators)
    spec.setdefault("ema_50", {"fn": "ema", "args": {"n": 50}})
    spec.setdefault("ema_200", {"fn": "ema", "args": {"n": 200}})
    spec.setdefault("atr_14", {"fn": "atr", "args": {"n": 14}})
    work = indicators.add_indicators(df, spec)
    if "donch_high_20" in (getattr(strategy, "required_indicators", {}) or {}) or \
            getattr(strategy, "name", "") == "breakout_donchian":
        work["donch_high_20"] = work["high"].rolling(20, min_periods=20).max().shift(1)
        work["donch_low_20"] = work["low"].rolling(20, min_periods=20).min().shift(1)
    work = indicators.classify_regime(work)

    rows = list(work.itertuples(index=False))
    open_trades: list[tuple[BacktestTrade, int]] = []
    closed: list[BacktestTrade] = []
    equity_curve: list[float] = []
    cur_equity = bankroll_usd
    peak = cur_equity
    max_dd = 0.0

    for i, row in enumerate(rows):
        ts = int(row.open_time)
        still_open: list[tuple[BacktestTrade, int]] = []
        for t, opened_at in open_trades:
            future = rows[opened_at + 1 : opened_at + 1 + t.horizon_bars + 1]
            tuples = [(int(r.open_time), float(r.open), float(r.high), float(r.low), float(r.close))
                      for r in future]
            timeout_ts = t.entry_ts + t.horizon_bars * tf_ms
            if not tuples or tuples[-1][0] < min(ts, timeout_ts):
                still_open.append((t, opened_at))
                continue
            resolved = _resolve_one_in_replay(t, tuples, tf_ms, gcfg.paper_fee_bps, gcfg.paper_slippage_bps)
            if resolved.exit_ts is None:
                still_open.append((t, opened_at))
                continue
            closed.append(resolved)
            cur_equity += resolved.pnl_usd or 0.0
            peak = max(peak, cur_equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - cur_equity) / peak)
        open_trades = still_open

        feats = {}
        for c in work.columns:
            if c in {"open_time", "open", "high", "low", "close", "volume", "regime"}:
                continue
            v = getattr(row, c, None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                feats[c] = float(v)
        regime = getattr(row, "regime", None)
        if not isinstance(regime, str):
            regime = None
        snap = Snapshot(
            symbol=symbol, timeframe=timeframe, ts=ts,
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume),
            indicators=feats, regime=regime,
        )
        cfg_for = cfg_strategy.for_regime(regime)
        sig = strategy.evaluate(snap, cfg_for, cost_bps)
        if sig is not None:
            t = BacktestTrade(
                entry_ts=ts, entry_price=sig.entry_price,
                side=sig.side, size_usd=sig.size_usd,
                tp_price=sig.tp_price, sl_price=sig.sl_price,
                horizon_bars=sig.horizon_bars,
                pred_p_up=sig.pred_p_up, edge=sig.edge,
                strategy=strategy.name, regime=regime,
                reason=sig.reason, estimator=sig.estimator,
            )
            open_trades.append((t, i))
        equity_curve.append(cur_equity)

    for t, opened_at in open_trades:
        future = rows[opened_at + 1 : opened_at + 1 + t.horizon_bars + 1]
        if not future:
            continue
        tuples = [(int(r.open_time), float(r.open), float(r.high), float(r.low), float(r.close))
                  for r in future]
        resolved = _resolve_one_in_replay(t, tuples, tf_ms, gcfg.paper_fee_bps, gcfg.paper_slippage_bps)
        if resolved.exit_ts is None:
            continue
        closed.append(resolved)
        cur_equity += resolved.pnl_usd or 0.0
        peak = max(peak, cur_equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - cur_equity) / peak)

    n = len(closed)
    won = sum(1 for t in closed if t.exit_reason == "TP")
    lost = sum(1 for t in closed if t.exit_reason == "SL")
    timed_out = sum(1 for t in closed if t.exit_reason == "TIMEOUT")
    wr = won / n if n else 0.0
    wlb, wub = wilson_interval(won, n)
    gross_pnl = sum((t.pnl_usd or 0) + (t.fee_usd or 0) + (t.slippage_usd or 0) for t in closed)
    fee_paid = sum(t.fee_usd or 0 for t in closed)
    slip_paid = sum(t.slippage_usd or 0 for t in closed)
    net = sum(t.pnl_usd or 0 for t in closed)
    holds = [t.bars_held or 0 for t in closed]
    avg_hold = sum(holds) / len(holds) if holds else 0.0
    pnls = np.array([t.pnl_usd or 0 for t in closed], dtype=float) if closed else np.array([])
    sharpe = float(pnls.mean() / pnls.std(ddof=0)) * math.sqrt(252) if pnls.size and pnls.std(ddof=0) > 0 else 0.0
    avg_b = 0.0
    if closed:
        ratios = []
        for t in closed:
            risk = abs(t.entry_price - t.sl_price)
            reward = abs(t.tp_price - t.entry_price)
            if risk > 0:
                ratios.append(reward / risk)
        if ratios:
            avg_b = sum(ratios) / len(ratios)
    be = break_even_win_rate(avg_b, cost_bps)
    if n >= 1000 and wlb > be:
        verdict = "edge_confirmed"
    elif n >= 1000 and wub < be:
        verdict = "no_edge"
    else:
        verdict = "inconclusive"
    return BacktestResult(
        strategy=strategy.name, symbol=symbol, timeframe=timeframe,
        start=int(work["open_time"].iloc[0]) if len(work) else 0,
        end=int(work["open_time"].iloc[-1]) if len(work) else 0,
        n_trades=n, n_won=won, n_lost=lost, n_timeout=timed_out,
        win_rate=wr, win_rate_wilson_lower=wlb, win_rate_wilson_upper=wub,
        gross_pnl=float(gross_pnl), fee_paid=float(fee_paid), slippage_paid=float(slip_paid),
        net_pnl=float(net), sharpe=sharpe, max_drawdown_pct=float(max_dd),
        avg_holding_bars=avg_hold, break_even_win_rate=be, verdict=verdict,
        trades=[asdict(t) for t in closed],
    )


class WalkForward:
    def __init__(self, train_window: int, test_window: int, step: int):
        self.train = train_window
        self.test = test_window
        self.step = step

    def windows(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        n = len(df)
        i = 0
        while i + self.train + self.test <= n:
            yield (df.iloc[i:i + self.train].reset_index(drop=True),
                   df.iloc[i + self.train:i + self.train + self.test].reset_index(drop=True))
            i += self.step


def run_backtest(
    strategy_name: str, df: pd.DataFrame, symbol: str, timeframe: str,
    bankroll_usd: float | None = None,
) -> BacktestResult:
    strat = get_strategy(strategy_name)
    gcfg = config.load()
    scfg = config.profile(gcfg.profile)
    bk = bankroll_usd if bankroll_usd is not None else scfg.bankroll_usd
    return simulate(strat, df, symbol, timeframe, scfg, gcfg, bk)


def save_result(result: BacktestResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = result.to_dict()
    d["trades"] = d["trades"][:5000]
    path.write_text(json.dumps(d, default=str, indent=2), encoding="utf-8")
