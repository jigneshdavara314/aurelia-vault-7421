from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from btcbot import (
    bankroll,
    backtest,
    calibration,
    config,
    data as data_mod,
    discover,
    indicators,
    predictions,
    resolver,
    self_improve,
    store,
    strategies as strat_mod,
)
from btcbot.errors import BtcBotError


def cmd_init(args) -> int:
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    active = config.active_strategies()
    per_strat = config.load().initial_deposit / max(1, len(active))
    for name in strat_mod.names():
        deposit = per_strat if name in active else 0.0
        bankroll.init_bankroll(strategy=name, mode="PAPER",
                               initial_deposit=deposit if deposit > 0 else 1.0)
    print("initialized.")
    print(f"  db: {config.DB_PATH}")
    print(f"  total paper bankroll: ${config.load().initial_deposit:.2f}")
    print(f"  active strategies: {active}  (${per_strat:.2f} each)")
    print(f"  registered (inactive): {[n for n in strat_mod.names() if n not in active]}")
    return 0


def cmd_status(args) -> int:
    cfg = config.load()
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    for name in strat_mod.names():
        bankroll.init_bankroll(strategy=name, mode="PAPER")
    s = bankroll.aggregate_summary(mode="PAPER")
    open_n = store.open_position_count()
    print(f"mode={cfg.mode}  symbol={cfg.symbol}  tf={cfg.timeframe}")
    print(f"balance=${s.get('balance',0)}  exposure=${s.get('open_exposure',0)}"
          f"  equity=${s.get('total_equity',0)}")
    print(f"profit=${s.get('profit',0)}  return={s.get('return_pct',0)*100:.2f}%"
          f"  drawdown={s.get('drawdown_pct',0)*100:.2f}%  halted={s.get('drawdown_halted')}")
    print(f"open positions: {open_n}")
    print(f"active strategies: {config.active_strategies()}")
    print(f"daily budget: ${config.daily_budget():.2f}")
    return 0


def cmd_download(args) -> int:
    loader = data_mod.BinanceVisionLoader()
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until) if args.until else None
    out = loader.ensure_history(args.symbol, args.timeframe, start, end)
    df = loader.load(args.symbol, args.timeframe)
    tf_ms = config.timeframe_ms(args.timeframe)
    gaps = loader.detect_gaps(df, tf_ms)
    print(f"parquet: {out}")
    print(f"rows: {len(df)}")
    print(f"gaps: {len(gaps)}")
    if gaps[:5]:
        print(f"  first gaps: {gaps[:5]}")
    return 0


def cmd_backtest(args) -> int:
    loader = data_mod.BinanceVisionLoader()
    df = loader.load(args.symbol, args.timeframe)
    if args.start:
        ts0 = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
        df = df[df["open_time"] >= ts0]
    if args.end:
        ts1 = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp() * 1000)
        df = df[df["open_time"] < ts1]
    df = df.reset_index(drop=True)
    if df.empty:
        print("no data in window")
        return 2
    res = backtest.run_backtest(args.strategy, df, args.symbol, args.timeframe)
    out_path = config.ROOT / f"backtest_{args.strategy}.json"
    backtest.save_result(res, out_path)
    print(f"strategy={res.strategy}  n={res.n_trades}  WR={res.win_rate:.3f}"
          f"  WLB={res.win_rate_wilson_lower:.3f}  WUB={res.win_rate_wilson_upper:.3f}")
    print(f"net_pnl=${res.net_pnl:.2f}  fees=${res.fee_paid:.2f}  slip=${res.slippage_paid:.2f}")
    print(f"sharpe={res.sharpe:.3f}  max_dd={res.max_drawdown_pct*100:.2f}%"
          f"  avg_hold={res.avg_holding_bars:.1f} bars")
    print(f"break-even WR={res.break_even_win_rate:.3f}  verdict={res.verdict}")
    print(f"saved: {out_path}")
    return 0


def cmd_trade(args) -> int:
    from serve import tick_trade
    tick_trade()
    return 0


def cmd_resolve(args) -> int:
    from btcbot import exchange as ex_mod
    ex = ex_mod.Exchange(config.load().exchange)
    out = resolver.resolve_all_open(ex)
    for r in out:
        print(r)
    return 0


def cmd_history(args) -> int:
    since_days = int(args.since.rstrip("d") or "7")
    since_ts = config.time_now_ms() - since_days * 86_400_000
    trades = store.query_trades({"since_ts": since_ts}, limit=500)
    print(f"{'id':>5} {'strategy':<22} {'side':<5} {'status':<8} {'entry':>10} {'exit':>10} {'pnl':>9}")
    for t in trades:
        ex_p = t.get("exit_price")
        pnl = t.get("pnl_usd")
        print(f"{t['id']:>5} {t['strategy']:<22} {t['side']:<5} {t['status']:<8} "
              f"{t['entry_price']:>10.2f} "
              f"{ex_p if ex_p is None else f'{ex_p:>10.2f}'} "
              f"{pnl if pnl is None else f'{pnl:>9.2f}'}")
    return 0


def cmd_report(args) -> int:
    print("=" * 60)
    print("aggregate")
    print(json.dumps(bankroll.aggregate_summary(mode="PAPER"), indent=2))
    if args.by_strategy:
        for name in strat_mod.names():
            s = bankroll.summary(strategy=name, mode="PAPER")
            perf = store.performance_summary(strategy=name)
            print("-" * 60)
            print(f"strategy: {name}")
            print(json.dumps({**s, **perf}, indent=2))
    return 0


def cmd_calibration(args) -> int:
    since_days = int((args.since or "90d").rstrip("d"))
    since_ts = config.time_now_ms() - since_days * 86_400_000
    trades = store.query_trades({"since_ts": since_ts, "strategy": args.strategy}, limit=10_000)
    diag = calibration.reliability_diagram(trades, slice_by=args.by)
    print(json.dumps({"strategy": args.strategy, "diagram": diag,
                      "verdict": calibration.verdict_for(diag)}, indent=2))
    return 0


def cmd_self_improve(args) -> int:
    out = self_improve.run(config.time_now_ms())
    print(json.dumps(out, indent=2))
    return 0


def cmd_export_snapshots(args) -> int:
    from btcbot import exchange as ex_mod
    cfg = config.load()
    ex = ex_mod.Exchange(config.load().exchange)
    n = int(args.n or 200)
    df = ex.fetch_recent_candles(cfg.symbol, cfg.timeframe, n + 300, drop_unclosed=True)
    spec = {
        "ema_50": {"fn": "ema", "args": {"n": 50}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "z_20": {"fn": "zscore", "args": {"n": 20}},
        "rsi_14": {"fn": "rsi", "args": {"n": 14}},
    }
    df = indicators.add_indicators(df, spec)
    df = indicators.classify_regime(df)
    snaps = list(data_mod.replay(df, cfg.symbol, cfg.timeframe, indicator_spec=None, warmup_bars=300))[-n:]
    prompt = (config.ROOT / "prompts" / "predict_direction.md")
    template = prompt.read_text(encoding="utf-8") if prompt.exists() else None
    path = predictions.export_snapshots_todo(snaps, prompt_template=template)
    print(f"exported {len(snaps)} snapshots to {path}")
    if template:
        print(f"prompt template at {prompt}")
    return 0


def cmd_backup(args) -> int:
    import shutil
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = config.BACKUPS_DIR / f"trades-{ts}.db"
    shutil.copy2(config.DB_PATH, out)
    print(f"backed up: {out}")
    return 0


def cmd_snapshot_day(args) -> int:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d") if not args.day else args.day
    store.save_daily_snapshot(day, config.time_now_ms())
    print(f"snapshot saved for {day}")
    return 0


def cmd_reconcile_live(args) -> int:
    print("LIVE reconciliation requires live_enabled=true and Phase 9 wiring.")
    print(f"live_enabled: {config.is_live_enabled()}")
    return 0


def cmd_discover(args) -> int:
    cfg = config.load()
    out = discover.run(cfg.symbol, cfg.timeframe)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_variants(args) -> int:
    rows = discover.list_variants()
    if not rows:
        print("no variants discovered yet")
        return 0
    print(f"{'variant':<48} {'tier':<10} {'streak':>6} {'n':>5} {'WR':>6} {'WLB':>6} {'BE':>6} {'NET':>8}")
    for r in rows:
        wr = f"{(r['last_wr'] or 0)*100:5.1f}%" if r['last_wr'] is not None else "  -  "
        wlb = f"{(r['last_wlb'] or 0)*100:5.1f}%" if r['last_wlb'] is not None else "  -  "
        be = f"{(r['last_be'] or 0)*100:5.1f}%" if r['last_be'] is not None else "  -  "
        net = f"${r['last_net']:>7.2f}" if r['last_net'] is not None else "    -   "
        print(f"{r['variant']:<48} {r['tier']:<10} {r['pass_streak']:>6} "
              f"{(r['last_n'] or 0):>5} {wr:>6} {wlb:>6} {be:>6} {net:>8}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="btcbot")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("status")

    sp = sub.add_parser("download")
    sp.add_argument("--symbol", default="BTC/USDT")
    sp.add_argument("--timeframe", default="5m")
    sp.add_argument("--since", required=True)
    sp.add_argument("--until", default=None)

    sp = sub.add_parser("backtest")
    sp.add_argument("--strategy", required=True)
    sp.add_argument("--symbol", default="BTC/USDT")
    sp.add_argument("--timeframe", default="5m")
    sp.add_argument("--start", default=None)
    sp.add_argument("--end", default=None)

    sub.add_parser("trade")
    sub.add_parser("resolve")

    sp = sub.add_parser("history")
    sp.add_argument("--since", default="7d")

    sp = sub.add_parser("report")
    sp.add_argument("--by-strategy", action="store_true", dest="by_strategy")

    sp = sub.add_parser("calibration")
    sp.add_argument("--strategy", required=True)
    sp.add_argument("--by", default="regime")
    sp.add_argument("--since", default="90d")

    sub.add_parser("self-improve")

    sp = sub.add_parser("export-snapshots")
    sp.add_argument("--n", default="200")

    sub.add_parser("backup")
    sp = sub.add_parser("snapshot-day")
    sp.add_argument("--day", default=None)

    sub.add_parser("reconcile-live")
    sub.add_parser("discover")
    sub.add_parser("variants")

    args = p.parse_args(argv)
    handlers = {
        "init": cmd_init, "status": cmd_status, "download": cmd_download,
        "backtest": cmd_backtest, "trade": cmd_trade, "resolve": cmd_resolve,
        "history": cmd_history, "report": cmd_report,
        "calibration": cmd_calibration, "self-improve": cmd_self_improve,
        "export-snapshots": cmd_export_snapshots, "backup": cmd_backup,
        "snapshot-day": cmd_snapshot_day, "reconcile-live": cmd_reconcile_live,
        "discover": cmd_discover, "variants": cmd_variants,
    }
    try:
        return handlers[args.cmd](args)
    except BtcBotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
