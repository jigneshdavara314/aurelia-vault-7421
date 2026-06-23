# AITradeCenter — Full Plan

A 12-month roadmap for a Bitcoin paper-trading bot, modeled on the
`polymarket-ai` architecture, scaled up for the realities of crypto markets.

> **Paper-trading only.** No real-money execution path will be wired up until
> we can show, with statistical honesty, that a strategy has positive expected
> value net of fees and slippage. The plan is structured so that every phase
> ends with a kill-criterion: if the measurement says "no edge," we do not
> advance to the next phase.

Author / operator: solo developer on Windows + Python. Reference codebase:
`C:\wamp\www\polymarket-ai\polybot\` (already studied end-to-end).

---

## 0. Operating principles (apply to every phase)

These are the rules the whole project lives under. They are non-negotiable.

1. **Honest measurement over hopeful narrative.** Every win-rate, every PnL
   number, every "this looks good" gets a Wilson confidence interval and a
   net-of-cost calculation. We treat `lower bound > 0` as the bar, not
   `point estimate > 0`.
2. **Paper has to bleed like live would.** Fees and slippage are applied in
   paper mode from day one. If you can't make paper profitable with realistic
   costs, you definitely can't make live profitable.
3. **One source of truth = `trades.db`.** Settlements, bankroll moves, and
   resolutions are written in a single atomic SQLite transaction. No JSON
   files holding load-bearing financial state. Audit log appended, never
   rewritten.
4. **Drop the unclosed candle.** The #1 silent backtest-vs-live divergence
   bug. Signal generation always operates on the last *closed* candle.
5. **Same code in backtest, paper, and (eventual) live.** The strategy class
   takes a `Snapshot` and returns a `Signal`. The harness around it changes;
   the strategy does not.
6. **Drawdown halt with peak ratchet.** Hard kill switch from polybot, carried
   over verbatim. The bot stops trading if equity falls more than
   `DRAWDOWN_HALT_FRAC` from peak. Manual reset only.
7. **Kill criteria at every phase boundary.** Each phase has explicit
   "advance" and "do not advance" thresholds defined upfront, so we can't
   rationalize forward later.
8. **No live trading path is built until Phase 9.** And even then it ships
   disabled-by-default with a fail-closed funds check.
9. **No comments unless the *why* is non-obvious.** Same discipline polybot
   uses. Code reads itself.
10. **No promises of profit.** This document never says "this will make
    money." It says "this will measure whether an idea has edge."

---

## 1. Glossary (so later sections are unambiguous)

- **Snapshot** — `(symbol, ts, ohlcv, indicators, regime)` at a single bar.
  The BTC analog of polybot's `Market` object.
- **Signal** — strategy output: `(snapshot, side, pred_p_up, edge, size_usd,
  tp_price, sl_price, horizon_bars, reason, estimator)`.
- **Position** — an open trade. Has `entry_price`, `tp_price`, `sl_price`,
  `timeout_ts`, `size_usd`, `side`. Closed by the resolver.
- **Resolve** — close a position because TP, SL, or timeout fired. Analog of
  polybot's `cmd_resolve`.
- **Cell** — a `(strategy, regime, indicator_band, side)` tuple. The unit
  `self_improve.py` promotes, demotes, or disables. Direct port of polybot.
- **Regime** — a classifier output: `{trending_up, trending_down, ranging,
  high_vol, low_vol, news_shock}`. Inferred from indicators, not predicted.
- **Edge** — `pred_p_up − 0.5 − cost_bps/10000` for directional bets. Must
  clear `min_edge`. Mirrors polybot's `fair − market − min_edge`.
- **Kelly size** — `f* = (b*p − q)/b * kelly_fraction`, where `b =
  tp_ret/sl_ret`. Capped by `max_position_usd` and `fillable_depth`.
- **Cost model** — `total_cost_bps = fee_bps + half_spread_bps +
  slippage_bps`. Applied at entry and exit. Configurable per venue.
- **Walk-forward** — backtest discipline where train and test windows roll
  forward in time, no look-ahead. The only honest backtest shape.

---

## 2. Project skeleton (final target shape)

```
f:\AITradeCenter\
├── PLAN.md                       <- this file
├── README.md                     written in Phase 1
├── settings.json                 daily_budget_usd, symbol, timeframe, profile
├── trades.db                     SQLite system of record
├── predictions.jsonl             Claude-in-loop fair direction (Phase 5+)
├── strategy_state.json           atomic-written per-cell tuner state (Phase 7)
├── self_improve_log.jsonl        append-only audit (Phase 7)
├── edge_scan_history.jsonl       daily recurrence proof (Phase 7)
├── data/
│   ├── binance_vision/           bulk historical ZIPs cached on disk
│   └── parquet/                  normalized per-symbol/timeframe parquets
├── btcbot/                       importable package
│   ├── __init__.py
│   ├── config.py                 dataclasses, profiles, env, paths
│   ├── exchange.py               ccxt wrapper: candles, orderbook, depth
│   ├── data.py                   Snapshot dataclass + bulk loader + replay
│   ├── indicators.py             pandas EMA/RSI/ATR/zscore/regime classifier
│   ├── strategy.py               evaluate(snapshot) → Signal + Kelly sizing
│   ├── strategies/               one file per strategy
│   │   ├── __init__.py
│   │   ├── base.py               Strategy ABC
│   │   ├── momentum.py
│   │   ├── nsigma_fade.py
│   │   ├── breakout.py
│   │   ├── funding_arb.py        Phase 8
│   │   └── claude_pred.py        Phase 5
│   ├── engine.py                 gate chain + should_attempt
│   ├── executor.py               PAPER (slippage+fee), LIVE disabled scaffold
│   ├── store.py                  SQLite: trades, bankroll, snapshots, atomic
│   ├── bankroll.py               compounding, drawdown halt, exposure_ok
│   ├── resolver.py               per-bar TP/SL/timeout loop
│   ├── backtest.py               walk-forward replay
│   ├── calibration.py            reliability diagram + Wilson CIs
│   ├── self_improve.py           per-cell trial/exploratory/disable ladder
│   └── predictions.py            predictions.jsonl loader
├── run.py                        CLI dispatcher
├── dashboard.py                  Flask read-side
├── serve.py                      cron-style scheduler for trade+resolve
├── tests/                        unit + property tests
└── notebooks/                    one-off research (not deployed)
```

---

## 3. Phase plan (12 months, kill-criteria included)

Months are budget, not deadlines. If a phase needs more time we slow down.
If it needs less, we slow down anyway and do more validation before
advancing.

### Phase 0 — Foundations (Week 1)

**Goal:** make the project runnable on this Windows machine end-to-end with
zero strategy logic, just plumbing.

Tasks:
- Create the directory skeleton from §2.
- Pin Python version (3.11+), create `pyproject.toml` or `requirements.txt`.
- Install: `ccxt`, `pandas`, `numpy`, `pyarrow`, `flask`, `pytest`,
  `python-dotenv`, `anthropic` (later phases), `pandas-ta` or `ta-lib-binary`.
- Set up SQLite WAL mode, basic schema migrations table.
- Write `btcbot/config.py` with `StrategyConfig` dataclass, three profiles
  (conservative / moderate / aggressive), `.for_regime()` overrides skeleton,
  `MODE`, `DAILY_BUDGET_USD`, `DRAWDOWN_HALT_FRAC`, `AGG_EXPOSURE_FRAC`,
  `PAPER_SLIPPAGE_BPS`, `PAPER_FEE_BPS`, `DB_PATH`, paths.
- Write `btcbot/store.py` schema bootstrap (`init_db()`) + `bankroll.py`
  initial-deposit seed.
- Write the smallest possible `run.py` that prints `bankroll.summary()`.
- Write `tests/test_smoke.py` that imports the package, initializes the DB,
  and asserts an empty bankroll.

**Done when:** `python run.py status` runs without error on a fresh checkout
and shows `balance=500.00, positions=0`.

**Kill criteria:** none — this phase is pure plumbing. If it doesn't run,
fix it.

---

### Phase 1 — Data layer (Weeks 2-3)

**Goal:** own years of clean BTC OHLCV history offline, plus a reliable live
candle feed.

Tasks:
- `btcbot/exchange.py`:
  - `class Exchange` wrapping `ccxt.binance({'enableRateLimit': True})`.
  - `fetch_recent_candles(symbol, timeframe, n)` — drops the unclosed last
    candle; returns a pandas DataFrame with UTC epoch ms `open_time`.
  - `fetch_orderbook(symbol, depth=20)` — top-of-book + L2 slice.
  - `fillable_depth(symbol, side, max_slippage_bps)` — walks the L2 book and
    returns USD size fillable within the slippage budget.
  - `set_sandbox_mode(True)` switch but with a comment that Binance sandbox
    is not usable for our data path — we use prod read endpoints only.
- `btcbot/data.py`:
  - `Snapshot` dataclass: `symbol, ts, open, high, low, close, volume,
    indicators: dict, regime: str | None`.
  - `BinanceVisionLoader`: downloads monthly ZIPs from `data.binance.vision`,
    caches under `data/binance_vision/`, parses into a single parquet per
    `(symbol, timeframe)` under `data/parquet/`. Idempotent — re-runs
    re-fetch only missing months.
  - `replay(symbol, timeframe, start, end) → Iterator[Snapshot]` for the
    backtest harness.
  - Gap detection: if `ts[i+1] - ts[i] != tf_ms`, log and skip — never
    silently iterate over gaps.
- `tests/test_data.py`:
  - Property: every snapshot's `ts` is a multiple of `tf_ms`.
  - Property: no duplicated timestamps in the loader output.
  - Property: forward-only — `ts[i+1] > ts[i]` for all i.
  - Reads a small fixture parquet and asserts round-trip integrity.

**Done when:**
- `python run.py download --symbol BTC/USDT --timeframe 5m --since 2022-01-01`
  produces a ~10M-row parquet with no gaps detected.
- `python run.py download` rerun is a no-op.
- `Exchange.fetch_recent_candles('BTC/USDT', '5m', 100)` returns 100 rows,
  none of which is the unclosed current candle.

**Kill criteria:** if Binance Vision is unavailable for some reason, fall
back to looping `fetch_ohlcv` with `since=` pagination — but log a warning
because the bulk archive is faster and free. Either way, history coverage is
mandatory before Phase 2.

---

### Phase 2 — Indicators + regime classifier (Week 4)

**Goal:** decorate every snapshot with the indicators strategies need, and
attach a regime label so we can later condition strategies on regime.

Tasks:
- `btcbot/indicators.py`:
  - Vectorized pandas functions: `ema(n)`, `sma(n)`, `rsi(n)`, `atr(n)`,
    `bollinger(n, k)`, `zscore(n)`, `realized_vol(n)`, `donchian(n)`,
    `vwap`, `volume_zscore(n)`.
  - `add_indicators(df, spec: dict) → df` — declarative spec so strategies
    can name what they need.
  - `classify_regime(df) → df['regime']` — heuristic, not predictive:
    - `trending_up` if `close > ema_200` and `ema_50.slope > 0`
    - `trending_down` symmetric
    - `ranging` if `atr_pct < median(atr_pct, 200)` and price within
      Bollinger bands
    - `high_vol` if `realized_vol_24h > p90(realized_vol_24h, 90d)`
    - else `mixed`
  - All indicators must be **strictly past-only** — write a property test
    that asserts no future leakage by re-running the indicator on a prefix
    of the series and checking values match.
- `tests/test_indicators.py`:
  - Leak test: `add_indicators(df[:n])[-1] == add_indicators(df)[n-1]` for
    every column, for many n.
  - Sanity: `rsi` stays in [0, 100], `atr >= 0`, regime is one of the known
    labels.

**Done when:** loading a parquet, adding indicators + regime, and saving back
takes < 5 seconds for 1 year of 5m candles, and the leak test passes.

**Kill criteria:** none — this is infrastructure. But if the leak test
fails, we stop and fix before any strategy is allowed to read indicators.

---

### Phase 3 — First strategy + honest backtest (Weeks 5-7)

**Goal:** one strategy, walk-forward backtested with realistic costs, with a
Wilson-CI verdict.

Tasks:
- `btcbot/strategies/base.py`: `Strategy` ABC with:
  - `name: str`
  - `required_indicators: dict`
  - `evaluate(snapshot) → Signal | None`
- Pick ONE strategy for Phase 3. Recommendation: **N-sigma fade**, because
  it's the closest analog of polybot's longshot-fade and the literature
  shows mean-reversion has *some* edge intraday before fees.
  - Rules: if `zscore_20 < -2` and regime in `{ranging, low_vol}`, go LONG
    with `tp = entry + 1*atr`, `sl = entry - 0.7*atr`, horizon 12 bars.
  - Symmetric short rule when permitted (Phase 3 is spot, so LONG-only).
- `btcbot/strategy.py`:
  - `kelly_size(p, b, cfg)` — `b = tp_ret/sl_ret`, returns USD stake.
  - `cost_model(symbol, size_usd, side) → bps` — fee + half-spread +
    expected slippage from `fillable_depth`.
  - `evaluate(snapshot, strategy) → Signal | None`.
- `btcbot/backtest.py`:
  - `WalkForward(train_window, test_window, step)` — rolls forward in time,
    yields `(train_df, test_df)` pairs.
  - `simulate(strategy, test_df, cost_model, paper_slippage_bps,
    paper_fee_bps) → BacktestResult`:
    - Iterates bar-by-bar.
    - On signal, opens a virtual position. On every subsequent bar, checks
      TP/SL/timeout against the *bar's* high/low — and applies the same
      "TP or SL first?" disambiguation freqtrade uses (assume SL first when
      both are in range; this is the conservative default).
    - Closes; books `pnl = (exit - entry) * size - 2*fee - 2*slippage`.
    - Stores every trade in the result for later calibration.
  - `BacktestResult`: `trades, n, win_rate, win_rate_wilson_lower,
    win_rate_wilson_upper, total_pnl, sharpe, max_drawdown, avg_holding_bars,
    verdict`.
- `tests/test_backtest.py`:
  - Synthetic data with a known signal: backtest returns the expected
    number of trades and positive PnL.
  - Cost model: a strategy with zero edge but realistic costs produces
    negative PnL (sanity).
- CLI: `python run.py backtest --strategy nsigma_fade --symbol BTC/USDT
  --timeframe 5m --start 2022-01-01 --end 2024-12-31`.

**Done when:**
- The walk-forward backtest produces a JSON report with n_trades,
  win-rate + Wilson CI, net PnL after fees + slippage, max drawdown.
- The synthetic-data tests pass.
- The cost-model sanity test passes (zero-edge strategy → losing).

**Kill criteria (the first real one):**
- If Wilson lower bound on win rate ≤ break-even win rate after costs over
  ≥ 1000 trades, we **iterate** on the strategy (different threshold,
  different regime gating) — but the bot does not advance to Phase 4 with
  this strategy. We try at most 3 iterations, then move on to a different
  strategy archetype (breakout, momentum) before declaring N-sigma-fade
  has no edge on BTC and burying it.

---

### Phase 4 — Paper-trading loop (Weeks 8-10)

**Goal:** the bot, running continuously, opening and closing paper positions
on live BTC data, with a real audit trail.

Tasks:
- `btcbot/store.py`:
  - Tables: `trades` (id, symbol, entry_ts, side, entry_price, size_usd,
    tp_price, sl_price, timeout_ts, status, exit_ts, exit_price, pnl_usd,
    fee_usd, slippage_usd, strategy, reason, estimator), `bankroll` (id=1,
    balance, peak_equity, initial_deposit, last_updated), `bankroll_log`
    (append-only), `daily_equity`, `daily_snapshots`.
  - Atomic `settle_and_credit(trade_id, exit_price, exit_reason, fee,
    slippage) → (pnl, new_balance)`. One transaction. Mirrors polybot.
  - `already_open(symbol, entry_window_ts)` for dedup.
- `btcbot/bankroll.py`:
  - `summary()`, `deduct_stake()`, `credit_payout()`, `peak_equity()`,
    `drawdown_halted()`, `exposure_ok(new_stake)`. Direct port.
- `btcbot/engine.py`:
  - Gate chain (each a `GateResult`-returning pure predicate):
    - `gate_already_open` — same `(symbol, entry_bar)` not double-opened.
    - `gate_position_cap` — `max_open_positions`.
    - `gate_per_symbol_cap` — at most N concurrent BTC positions.
    - `gate_daily_spend` — daily budget from `settings.json`.
    - `gate_exposure` — `bankroll.exposure_ok(stake)`.
    - `gate_drawdown_halt` — `not bankroll.drawdown_halted()`.
    - `gate_fillable_depth` — exchange has enough top-of-book to take this
      size within slippage budget.
    - `gate_affordable` — `bankroll.can_afford(stake)`.
  - `run_gates(...) → GateResult`, returns first failure.
  - `should_attempt(...)` — PAPER simulates fill with a stable per-UTC-day
    hash (same trick polybot uses); LIVE always attempts.
- `btcbot/executor.py`:
  - `class Executor` with `execute(signal) → result`.
  - `_execute_paper` — applies `PAPER_SLIPPAGE_BPS` to fill price, books
    fee, writes trade row, deducts bankroll.
  - `_execute_live` — present but raises `NotImplementedError` until Phase 9.
- `btcbot/resolver.py`:
  - `resolve_all_open(now_ts)` — for each open position, fetch the bars
    since `entry_ts`, check whether high/low hit TP or SL, or whether
    `now_ts >= timeout_ts`. Use the same intra-bar disambiguation as the
    backtest. Settle atomically.
  - Per-bar invariant: `resolve_all_open` is idempotent — running it twice
    for the same `now_ts` is a no-op.
- CLI:
  - `python run.py trade` — fetch latest candle, evaluate, gate, execute.
  - `python run.py resolve` — close anything that should be closed.
  - `python run.py status` — bankroll + open positions.
  - `python run.py history --since 7d` — closed trades.
- `serve.py` — simple scheduler (apscheduler or pure-Python loop) that runs
  `trade` then `resolve` every 5 minutes. Windows-friendly.
- `tests/test_engine.py`, `test_resolver.py`, `test_executor_paper.py`:
  - Resolver idempotency.
  - Atomic settle: simulated crash mid-settle leaves the DB consistent.
  - Drawdown halt prevents new trades.

**Done when:**
- `serve.py` runs for 72 hours unattended on this machine.
- Trades open and close in `trades.db`.
- `dashboard.py` placeholder displays bankroll equity curve.
- Process restart resumes with no double-fills, no orphan positions.

**Kill criteria:**
- If during the 72-hour paper run the bot enters significantly fewer trades
  than the backtest projected (e.g. < 30% of the expected rate), the gate
  chain or fillable_depth logic is too strict — iterate, don't advance.
- If paper P&L over 30 days of running is significantly worse than the
  backtest predicted on the same window (more than 2 standard deviations
  below) — investigate. Almost always the bug is "unclosed candle leaked
  into signal," fee tier mismatch, or slippage under-modeled.

---

### Phase 5 — Calibration + Claude-in-loop (Weeks 11-13)

**Goal:** stop trusting `pred_p_up` until we've measured whether it's
calibrated. Also: bring in an LLM as a *judge*, not a price oracle.

Tasks:
- `btcbot/calibration.py`:
  - `reliability_diagram(closed_trades_with_pred_p) → buckets`.
  - For each bucket of predicted probability (e.g. [0.50, 0.55, 0.60, ...]),
    compute realized win rate and Wilson CI.
  - A calibrated model has `realized ≈ predicted` per bucket. Miscalibration
    is the silent killer of Kelly sizing.
  - Per-regime and per-strategy slices.
- `btcbot/predictions.py`:
  - Reads `predictions.jsonl`: `{snapshot_id, pred_p_up, confidence,
    regime, rationale}`. Same shape as polybot's `estimates.json` but
    line-delimited and append-only.
  - `get_prediction(snapshot_id) → dict | None`.
- `btcbot/strategies/claude_pred.py`:
  - Exports a batch of "interesting" snapshots (recent N-sigma moves,
    regime transitions) to `predictions_todo.jsonl`.
  - When `predictions.jsonl` has a row for a snapshot, `evaluate` uses
    `pred_p_up` instead of the heuristic.
  - Falls back to the heuristic when no prediction is available, but
    *records* the estimator used per trade — so calibration can compare
    Claude vs heuristic per regime.
- `run.py export-snapshots` — produces `predictions_todo.jsonl` for the
  next Claude session.
- `run.py calibration --strategy claude_pred --since 90d` — prints the
  reliability diagram + Wilson CIs + a verdict ("overconfident in the
  0.60-0.65 bucket by 8pp," etc.).

**Done when:**
- After 30+ days of paper trading with mixed estimators, the calibration
  report cleanly slices realized vs predicted per bucket per regime.
- An LLM-judge round-trip is working end-to-end:
  `export-snapshots → human/Claude fills → trade reads → resolver settles
  → calibration measures`.

**Kill criteria:**
- If both heuristic and Claude estimators are equally miscalibrated, the
  problem is the strategy archetype, not the estimator. Iterate on Phase 3.
- If Claude is calibrated but adds no PnL over the heuristic (within Wilson
  CI), the estimator complexity isn't earned. Keep it for research, don't
  promote.

---

### Phase 6 — Multi-strategy tournament (Weeks 14-17)

**Goal:** stop guessing which strategy is best. Run several with separate
bankrolls and let live P&L pick winners. Direct port of polybot's
`strategies.py`.

Tasks:
- `btcbot/strategies.py` (tournament config):
  - `STRATEGIES = {"nsigma_fade": {...}, "breakout_donchian": {...},
    "momentum_ema_cross": {...}, "claude_directional": {...}, ...}`.
  - Each has `kind`, `params`, `tiers`, `blurb`, `initial_deposit`.
- Multi-book bankroll:
  - `bankroll` table grows a `strategy` column. Each strategy gets its own
    starting deposit.
  - All gates run per-strategy: drawdown halt halts only that strategy.
- Add 2 more strategies beyond Phase 3:
  - `breakout_donchian` — long on close above `donchian_high(20)`,
    sl=`atr`, tp=`2*atr`, regime-gated to `trending_up`.
  - `momentum_ema_cross` — long when `ema_50 > ema_200` and we re-enter
    after a pullback to `ema_50`.
- Per-strategy reporting:
  - `python run.py report --by-strategy` — equity curve, win rate, Sharpe,
    Wilson CI per strategy.
- Per-strategy backtest in `backtest.py` so we can sanity-check before
  enabling a new strategy in the tournament.

**Done when:**
- Three strategies running concurrently in paper, separate bankrolls.
- Dashboard shows a side-by-side equity curve.
- 60+ days of multi-strategy data accumulated.

**Kill criteria:**
- If after 60+ days *every* strategy is flat-or-negative after fees, the
  paper bot has correctly measured that we don't have edge on BTC with
  these archetypes. We either go back to Phase 3 with new archetypes, or
  pivot to Phase 8 ideas (funding-rate arb, basis trades) earlier.

---

### Phase 7 — Self-improvement ladder (Weeks 18-21)

**Goal:** the bot autonomously promotes cells that work and demotes cells
that don't, with bounded stake multipliers and a full audit log. Port of
polybot's `self_improve.py`.

Tasks:
- `btcbot/self_improve.py`:
  - State machine per cell (`strategy × regime × indicator_band × side`):
    - `trial` (0.25x stake) → `exploratory` (0.5x) → `confirmed` (1.0x).
    - Promotions require N consecutive days of `wilson_lower > 0` net of
      costs. Polybot uses 2/5; we'll start with 5/10 for crypto's higher
      noise floor.
    - Demotion: rolling 30-trade `wilson_lower < 0` → demote one tier or
      disable.
    - Disabled cells are surfaced in the dashboard but not traded.
  - `strategy_state.json` is the *only* mutable file outside `trades.db`,
    and it's atomic-written.
  - `self_improve_log.jsonl` — append-only audit of every promote/demote.
- `edge_scan_history.jsonl`:
  - Daily backtest-on-recent-30-days produces `{cell, n, win_rate,
    wilson_lower, recurrence_count}`.
  - A cell needs `recurrence_count >= 5` (i.e. it had edge in 5 of the last
    N daily scans) to be eligible for promotion.
- `python run.py self-improve` — the nightly job. Idempotent.

**Done when:**
- A cell that consistently wins gets auto-promoted from trial → exploratory
  → confirmed over weeks, with audit entries proving the journey.
- A cell that consistently loses gets disabled.
- `strategy_state.json` is never corrupted by a crash mid-write (verified
  by fuzz test).

**Kill criteria:**
- If the ladder promotes things that then immediately lose, the promotion
  thresholds are too loose for BTC noise. Tighten.
- If nothing ever promotes, the bar is too high. Loosen.
- This phase is calibration of the ladder itself — expect 2-3 rounds.

---

### Phase 8 — Beyond directional bets (Months 6-8)

**Goal:** add strategies that don't depend on predicting price direction,
because the research is clear that retail directional edge on BTC is at best
razor-thin.

Tasks:
- **Funding-rate arbitrage** (Phase 8a):
  - Requires futures (perp) data. Extend `exchange.py` with
    `fetch_funding_rate(symbol)`, `fetch_funding_history(symbol)`.
  - Strategy: when 8h funding is extremely positive (longs pay shorts),
    short the perp + buy spot of equal notional → collect funding.
  - Paper-model the two-leg execution carefully: each leg has its own
    slippage, fees are different (maker vs taker, spot vs perp).
  - Margin model: paper margin call simulation — if spot leg moves against
    you and perp leg gets liquidated, we want to *measure* that scenario.
- **Cash-and-carry basis trade** (Phase 8b):
  - Dated futures vs spot. Annualized yield. Lower-risk, lower-return.
  - Same two-leg accounting.
- **Mean-reversion grid in detected ranges** (Phase 8c):
  - Only active when regime classifier reports `ranging` for k consecutive
    days.
  - Place a ladder of buy-low / sell-high orders. Paper-execute.
  - Hard kill when regime exits `ranging`.
- Each is a new strategy in the tournament, on its own bankroll, with its
  own backtest.

**Done when:**
- At least one non-directional strategy survives Phase 7's ladder all the
  way to `confirmed` tier in paper.

**Kill criteria:**
- If funding arb backtests negative after fees + slippage (likely on
  smaller venues), drop it. The honest version of this strategy needs
  exchange-specific fee schedules baked in.

---

### Phase 9 — Live-mode scaffold, still disabled (Month 9)

**Goal:** wire up the live execution path so that flipping `MODE=LIVE`
*would* work — but it ships gated behind a manual code change and a
fail-closed funds check. We still do not trade live.

Tasks:
- `btcbot/executor.py::_execute_live`:
  - Real ccxt order placement against Binance.
  - Wait-for-fill polling with timeout.
  - Cancel-remainder on partial fill timeout.
  - Fail-closed: before placing, query account balance; if anything looks
    off (auth error, balance < expected, sandbox flag set), **refuse** to
    place.
- `MODE=LIVE` requires:
  - An environment variable explicitly set.
  - A `live_enabled = True` line in `settings.json` that the operator must
    manually add.
  - Both `BINANCE_API_KEY` and `BINANCE_API_SECRET` present.
  - All three: missing any → bot stays in PAPER and logs why.
- Live order recording: same `trades.db` schema, with `mode='LIVE'` flag,
  and the `bankroll` table is *separate* (`bankroll_live` row id=2) so we
  never confuse paper and live equity.
- Reconciliation: a separate `python run.py reconcile-live` that diffs
  exchange-side trade history against our DB and flags mismatches.

**Done when:**
- A mock live-mode test (sandbox account, tiny notional, fully manual) can
  place an order, fill, settle into `bankroll_live`, and reconcile.
- `MODE=PAPER` is still the default and the operator must take 3 explicit
  steps to enable live.

**Kill criteria:** if paper P&L over the 9 months has been flat or
negative *across the whole tournament*, we do not enable live trading.
The plan is to walk away from the live path, not to force it.

---

### Phase 10 — Dashboard + observability (Month 10)

**Goal:** see what the bot is doing without running CLI commands.

Tasks:
- `dashboard.py` — Flask app:
  - `/` overview: equity curves (per-strategy and aggregate), open
    positions, gate-failure stats over last 24h, drawdown bar.
  - `/trades` — sortable table of closed trades.
  - `/calibration` — reliability diagrams per strategy.
  - `/strategies` — current `strategy_state.json` view: per-cell tier,
    rolling win rate, Wilson CI, days at current tier.
  - `/regime` — regime history overlaid on price.
- Generates a static `dashboard.html` snapshot on demand for offline view.
- Optional: `notify.py` — Telegram or email notifier for:
  - Drawdown halt fired.
  - A cell got promoted to `confirmed`.
  - Daily summary at 00:00 UTC.

**Done when:** the dashboard is the primary way the operator checks the
bot.

**Kill criteria:** none — observability isn't optional.

---

### Phase 11 — Robustness, testing, audit (Month 11)

**Goal:** stop trusting the code and start *proving* it.

Tasks:
- Property tests:
  - Atomic settle: simulate crash-after-every-statement in
    `settle_and_credit`, assert DB never enters an inconsistent state.
  - Resolver idempotency: same input → same DB delta, n times.
  - Gate chain ordering: gates are commutative on `PASS`, short-circuit on
    first failure.
  - Walk-forward backtest determinism: same seed + same data + same
    strategy → same trades, byte-equal.
- Performance:
  - Profile the backtest: a 5-year 5m backtest should finish in minutes,
    not hours.
  - Profile the live loop: end-to-end `trade` invocation should fit in
    well under 60 seconds so the 5-minute schedule is safe.
- Data integrity:
  - Nightly job re-downloads the last 24h of candles and diffs against the
    DB-cached version. Discrepancies → alert.
- Recovery drills:
  - Simulate `trades.db` corruption — verify backup + restore works.
  - Simulate API outage — verify the bot logs and waits, doesn't loop-fail.
- External audit:
  - Hand the code to someone (or another agent) for review focused on
    "where could this lose me money silently."

**Done when:** the code has a test suite that catches the kinds of bugs
that historically cause "the bot was profitable in backtest but lost
money live."

**Kill criteria:** none — robustness isn't optional. But if the audit
surfaces fundamental issues, we stop and fix before moving on.

---

### Phase 12 — Decision month (Month 12)

**Goal:** look at 12 months of honest data and decide what to do.

Outputs:
- A `REPORT.md` that summarizes:
  - Cumulative paper P&L per strategy, net of fees + slippage.
  - Sharpe and max drawdown per strategy.
  - Wilson CIs on win rate.
  - Calibration quality per strategy.
  - Number of cells that survived the ladder.
  - Honest comparison to buy-and-hold BTC over the same period.
- Three possible decisions:
  1. **At least one strategy has Wilson_lower > 0 net of costs over 12
     months of paper trading.** Cautiously enable Phase 9 live mode with
     micro-notional (e.g. $10 trades) and monitor closely for one more
     month before scaling.
  2. **Mixed results — some strategies look promising but not at the
     Wilson_lower bar.** Iterate: more data, refined strategies, no live
     trading yet.
  3. **All strategies flat or negative.** The bot has done its job: it
     measured that we don't have edge with the archetypes we tried. Either
     try new archetypes (Phase 8 ideas, or new research) or close the
     project with the conclusion that retail BTC paper-to-live alpha is
     not extractable for us.

This phase is the most important one. It's the discipline that separates
this plan from a hopium project.

---

## 4. Cross-cutting concerns

### 4.1 Risk controls (apply continuously from Phase 4 onward)

- **Drawdown halt:** `DRAWDOWN_HALT_FRAC = 0.20` from peak equity. Manual
  reset only.
- **Per-strategy drawdown halt:** same, applied per-bankroll.
- **Exposure cap:** `AGG_EXPOSURE_FRAC = 0.50` — never more than half of
  bankroll deployed at once.
- **Per-symbol cap:** max N concurrent BTC positions (start N=3).
- **Daily spend cap:** `daily_budget_usd` from `settings.json`.
- **Position cap:** `max_open_positions = 20`.
- **Kelly fraction:** start at 0.25. Promote to 0.5 only if Phase 5
  calibration shows the model is well-calibrated for the bucket being
  sized.

### 4.2 Cost model honesty

Mirrors the research findings:
- Binance taker fee: 10 bps (7.5 bps with BNB discount, but we model the
  worse case in paper).
- Spread: half-spread bps from live orderbook at signal time.
- Slippage: derived from `fillable_depth` walking the L2 book.
- For backtests we use historical L1 spread estimates per regime (Phase 3
  uses a flat 5 bps, Phase 11 makes this regime-conditional).

### 4.3 Persistence and recovery

- `trades.db` is the system of record. Everything else is rebuildable.
- WAL mode on SQLite, `PRAGMA synchronous=NORMAL`.
- Nightly backup: copy `trades.db` to `backups/trades-YYYYMMDD.db`,
  keep 30 days.
- All JSON state (`strategy_state.json`, `predictions.jsonl`,
  `self_improve_log.jsonl`) is either atomic-written (write to tempfile +
  os.replace) or append-only.

### 4.4 Time and clock discipline

- Everything UTC, stored as epoch milliseconds.
- No `datetime.now()` without `tz=UTC`. Lint rule.
- Bar timestamps are open-time, not close-time. Carrying this convention
  end-to-end avoids a class of off-by-one bugs.

### 4.5 Configuration

- `settings.json` for live-tunable values (daily budget, kelly fraction,
  active strategies).
- Profiles in `config.py` for structural defaults.
- `.env` for secrets (Phase 9 only).
- No magic numbers in strategy code — every threshold is in
  `StrategyConfig` and per-strategy params.

### 4.6 Logging and audit

- One structured log file per day, JSONL.
- Every gate decision, every signal, every executed trade logged.
- The audit goal: given any closed trade, the operator can reconstruct
  exactly why the bot opened it, why it closed at the price it closed,
  what the bankroll state was at open and close, and what regime was
  active.

### 4.7 Backups and disaster recovery

- Nightly `trades.db` backup as above.
- Weekly: rsync `data/parquet/` to a second drive (or a cloud bucket if
  the operator chooses).
- Recovery drill once per quarter: nuke a local copy, restore from
  backup, verify `dashboard.py` shows the same equity curve.

### 4.8 Security

- Even in paper mode: no secrets in the repo, `.env` gitignored.
- Phase 9 onward: API keys are read-only + spot-trade only. No
  withdrawal permission ever.
- `executor._execute_live` short-circuits if the key has withdrawal
  permission. Fail-closed.

---

## 5. What we are explicitly NOT building

Naming these now so they don't sneak in:
- No reinforcement-learning agent. The research is unanimous that retail
  RL on BTC overfits the training set and dies live.
- No LSTM/transformer "price predictor" as a primary signal. Same reason.
- No hyperopt-style parameter sweep with `IntParameter`/`CategoricalParameter`.
  That's freqtrade's biggest curve-fitting trap.
- No high-frequency anything. Our cadence is 5 minutes, hard floor.
- No web UI beyond `dashboard.py`. No React, no FastAPI.
- No multi-asset support in Phase 1-7. BTC only. Adding alts is a
  *later-than-Phase-12* discussion.
- No DEX, no Uniswap-style execution. Centralized exchanges only.
- No copy-trading anyone else's signals. The bot has its own thesis or
  it doesn't trade.

---

## 6. Risks and what we'll do about them

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Backtest is profitable, paper is flat | High | Apply paper slippage + fees from day 1. Phase 5 calibration catches miscalibrated probs. Phase 11 audits drift. |
| Paper is profitable, live would be worse | High | We don't enable live until Phase 12 decision. Live ships micro-notional first. |
| Strategy works for 6 months, then market regime shifts and it dies | High | Regime classifier from Phase 2. Self-improvement ladder from Phase 7 demotes losing cells in 30 trades. Per-strategy drawdown halts. |
| Bug in `settle_and_credit` corrupts bankroll | Low (atomic) | Property tests in Phase 11 fuzz this. Nightly backup gives a known-good restore point. |
| Binance API changes / outage | Medium | Bot logs and waits, doesn't loop-fail. Phase 8 onward we may add a secondary exchange for resilience. |
| Operator gets impatient and enables live too early | High (human factor) | Phase 9 requires three explicit manual steps to enable live. The plan itself is the constraint. |
| 12 months of paper trading shows no edge | Realistic possibility | Phase 12 decision option 3 explicitly accepts this and exits gracefully. The infrastructure was the point. |
| LLM (Claude) hallucinates a directional call | Medium | Claude predictions are *measured* in calibration alongside the heuristic. If Claude is no better than coin-flip, we down-weight or disable it. |

---

## 7. Definition of done for the project as a whole

The project is "done" when, regardless of which Phase 12 branch fires:

- The codebase is reproducible: someone clones it, runs `python run.py
  download` and `python run.py serve`, and the bot trades.
- The decision report (`REPORT.md`) exists and is honest about what 12
  months of measurement showed.
- The next operator (future-you) can decide what to do with the
  infrastructure based on the report — extend, retire, or carefully
  graduate to live.

Done is not "the bot makes money." Done is "we know whether it can."
