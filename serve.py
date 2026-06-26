from __future__ import annotations

import logging
import sys

from btcbot import config


def tick_trade() -> None:
    from btcbot import bankroll, engine, exchange, executor, strategies, indicators
    from btcbot.data import Snapshot
    import math

    gcfg = config.load(force=True)
    ex = exchange.Exchange(gcfg.exchange)
    active = config.active_strategies()
    if not active:
        logging.info("no active strategies; skipping trade tick")
        return
    strat_objs = [strategies.get(n) for n in active]
    indicator_spec: dict = {}
    for s in strat_objs:
        indicator_spec.update(getattr(s, "required_indicators", {}) or {})
    indicator_spec.setdefault("ema_50", {"fn": "ema", "args": {"n": 50}})
    indicator_spec.setdefault("ema_200", {"fn": "ema", "args": {"n": 200}})
    indicator_spec.setdefault("atr_14", {"fn": "atr", "args": {"n": 14}})
    indicator_spec.setdefault("z_20", {"fn": "zscore", "args": {"n": 20}})
    df = ex.fetch_recent_candles(gcfg.symbol, gcfg.timeframe, 300, drop_unclosed=True)
    df = indicators.add_indicators(df, indicator_spec)
    df["donch_high_20"] = df["high"].rolling(20, min_periods=20).max().shift(1)
    df["donch_low_20"] = df["low"].rolling(20, min_periods=20).min().shift(1)
    df = indicators.classify_regime(df)
    if df.empty:
        return
    last = df.iloc[-1]
    feats = {c: float(last[c]) for c in df.columns
             if c not in {"open_time","open","high","low","close","volume","regime"}
             and last[c] == last[c]}
    regime = last["regime"] if "regime" in df.columns and isinstance(last["regime"], str) else None
    snap = Snapshot(
        symbol=gcfg.symbol, timeframe=gcfg.timeframe, ts=int(last["open_time"]),
        open=float(last["open"]), high=float(last["high"]),
        low=float(last["low"]), close=float(last["close"]),
        volume=float(last["volume"]), indicators=feats, regime=regime,
    )
    now_ts = config.time_now_ms()
    cost_bps = gcfg.paper_fee_bps + gcfg.paper_slippage_bps + 5
    scfg = config.profile(gcfg.profile)
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    from btcbot import store
    for s in strat_objs:
        bankroll.init_bankroll(strategy=s.name, mode="PAPER")
        sig = s.evaluate(snap, scfg.for_regime(regime), cost_bps)
        if sig is None:
            store.record_signal_event(now_ts, s.name, gcfg.symbol,
                                       "evaluated_no_signal", "")
            continue
        store.record_signal_event(now_ts, s.name, gcfg.symbol,
                                   "signal_emitted",
                                   f"side={sig.side} edge={sig.edge:.4f}")
        gate = engine.run_gates(sig, ex, now_ts)
        if not gate.ok:
            logging.info("gate blocked: %s %s", gate.gate, gate.reason)
            continue
        if not engine.should_attempt(sig, gcfg.mode, now_ts):
            logging.info("paper sim: no fill for %s", sig.snapshot.id)
            continue
        try:
            ex_obj = executor.Executor(ex)
            result = ex_obj.execute(sig)
            store.record_signal_event(now_ts, s.name, gcfg.symbol,
                                       "signal_filled",
                                       f"trade_id={result.get('trade_id')}")
            logging.info("trade opened: %s", result)
        except Exception as exc:
            logging.error("executor error: %s", exc)


def tick_resolve() -> None:
    from btcbot import exchange, resolver
    gcfg = config.load(force=True)
    ex = exchange.Exchange(gcfg.exchange)
    try:
        out = resolver.resolve_all_open(ex)
        for r in out:
            logging.info("resolved: %s", r)
    except Exception as exc:
        logging.error("resolve error: %s", exc)


def daily_jobs() -> None:
    import datetime as _dt
    from btcbot import store, self_improve, discover, patterns
    cfg = config.load()
    try:
        day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        store.save_daily_snapshot(day, config.time_now_ms())
        self_improve.run(config.time_now_ms())
        discover.run(cfg.symbol, cfg.timeframe)
        patterns.run(cfg.symbol, cfg.timeframe)
    except Exception as exc:
        logging.error("daily jobs error: %s", exc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load()
    logging.info("btcbot serve start mode=%s symbol=%s tf=%s", cfg.mode, cfg.symbol, cfg.timeframe)
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logging.error("apscheduler not installed; pip install apscheduler")
        return 1
    import datetime as _dt
    sched = BlockingScheduler(timezone="UTC")
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    sched.add_job(tick_trade, "interval", minutes=5, id="trade",
                  next_run_time=now_utc + _dt.timedelta(seconds=5))
    sched.add_job(tick_resolve, "interval", minutes=5, id="resolve",
                  next_run_time=now_utc + _dt.timedelta(seconds=35))
    sched.add_job(daily_jobs, "cron", hour=0, minute=5, id="daily")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
