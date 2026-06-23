# AITradeCenter — Full Development Plan

The implementation-level companion to `PLAN.md`. PLAN.md says *what and
when*; this file says *exactly how*. Every module's API, every DB column,
every gate, every test, every CLI command, every config knob.

A developer (or future-you) should be able to read this file top-to-bottom
and implement the project without making unsupervised design decisions.

> **Status convention.** Each subsection ends with a `STATE:` line:
> `STATE: spec` means the design is frozen but no code exists yet;
> `STATE: in-progress` means partial; `STATE: done` means
> implemented + tested. Start every item at `STATE: spec`.

---

## Table of contents

1. Environment + project bootstrap
2. Coding conventions
3. Configuration layer (`btcbot/config.py`, `settings.json`, `.env`)
4. Persistence schema (`trades.db`)
5. Exchange wrapper (`btcbot/exchange.py`)
6. Data layer (`btcbot/data.py`)
7. Indicators + regime (`btcbot/indicators.py`)
8. Strategy framework (`btcbot/strategy.py`, `btcbot/strategies/`)
9. Sizing model (Kelly + cost model)
10. Engine / gates (`btcbot/engine.py`)
11. Executor (`btcbot/executor.py`)
12. Resolver (`btcbot/resolver.py`)
13. Bankroll (`btcbot/bankroll.py`)
14. Store (`btcbot/store.py`)
15. Backtest harness (`btcbot/backtest.py`)
16. Calibration (`btcbot/calibration.py`)
17. Self-improvement ladder (`btcbot/self_improve.py`)
18. Predictions / LLM-in-loop (`btcbot/predictions.py`)
19. CLI (`run.py`)
20. Scheduler (`serve.py`)
21. Dashboard (`dashboard.py`)
22. Notifications (`notify.py`, optional)
23. Tests
24. Logging + audit
25. Backup + recovery
26. Performance budget
27. CI + automation
28. Live-mode wiring (Phase 9, dormant)
29. Operator runbook (day-to-day commands)
30. Decision report shape (Phase 12)

---

## 1. Environment + project bootstrap

### 1.1 Python + tooling

- Python **3.11** or **3.12**. Pinned via `pyproject.toml`.
- Package manager: `pip` + `requirements.txt` is enough. No poetry.
- Virtualenv at `f:\AITradeCenter\.venv\`.
- Windows-friendly throughout. Use forward slashes in code, raw strings
  for any Windows paths in tests.

### 1.2 Dependencies

Pin major versions; allow patch updates.

```
# core
ccxt>=4.4,<5
pandas>=2.2,<3
numpy>=1.26,<3
pyarrow>=16
python-dotenv>=1.0

# indicators
pandas-ta>=0.3.14b      # pure-Python, no ta-lib binary nightmare on Windows

# storage / serving
sqlalchemy>=2.0,<3
flask>=3
apscheduler>=3.10

# LLM (Phase 5+)
anthropic>=0.40

# dev
pytest>=8
pytest-cov>=5
hypothesis>=6
black>=24
ruff>=0.5
```

### 1.3 Bootstrap commands

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py init        # creates trades.db, settings.json defaults
python run.py status      # smoke test
```

STATE: spec.

---

## 2. Coding conventions

- **Formatter:** `black`, line length 100.
- **Linter:** `ruff`. Enable `E,F,W,B,UP,SIM,PL`.
- **Type hints:** required on every public function. `from __future__ import
  annotations` at the top of every module.
- **Dataclasses for data shapes**, not dicts. Use `@dataclass(frozen=True)`
  for immutable value types (Snapshot, Signal, GateResult).
- **No mutable module-level state** except for one explicit `_CONFIG`
  singleton populated by `config.load()`.
- **Time:** UTC everywhere. Functions that need "now" take `now_ts: int`
  (epoch ms) as a parameter for testability; only one `time_now_ms()`
  helper in `config.py` reads the real clock.
- **Money:** all USD values are `float` *until persisted*, then stored as
  `REAL` in SQLite. Bps values are integers.
- **No printing** in `btcbot/*`. Library code returns or raises. The CLI
  (`run.py`) and the scheduler (`serve.py`) print.
- **Errors:** raise typed exceptions from `btcbot/errors.py` —
  `ConfigError`, `DataError`, `ExchangeError`, `GateBlocked`, `StoreError`.
  Catch at the CLI boundary, not deeper.
- **Comments:** write none unless the *why* is non-obvious. No
  module-level docstrings longer than two lines.

STATE: spec.

---

## 3. Configuration layer

### 3.1 `btcbot/config.py`

Single source of truth for configuration. Reads `.env`, `settings.json`,
and exposes typed dataclasses.

```python
# Pseudocode-level signature list.

@dataclass(frozen=True)
class StrategyConfig:
    name: str
    min_edge: float              # required edge after costs, e.g. 0.02
    min_confidence: float        # minimum pred_p_up to enter, e.g. 0.55
    kelly_fraction: float        # 0.25 default
    max_position_usd: float      # hard cap per trade
    min_liquidity_usd: float     # fillable depth required
    max_spread_bps: int          # skip if spread wider than this
    min_price: float             # 0 for crypto, kept for symmetry
    max_price: float             # inf for crypto
    bankroll_usd: float          # current strategy bankroll (snapshot)
    regime_overrides: dict[str, dict]  # per-regime knob overrides

    def for_regime(self, regime: str) -> "StrategyConfig": ...

PROFILES: dict[str, StrategyConfig] = {
    "conservative": StrategyConfig(...),
    "moderate":     StrategyConfig(...),
    "aggressive":   StrategyConfig(...),
}

@dataclass
class GlobalConfig:
    mode: Literal["PAPER", "LIVE"]
    symbol: str
    timeframe: str                       # "5m"
    daily_budget_usd: float
    drawdown_halt_frac: float            # 0.20
    agg_exposure_frac: float             # 0.50
    paper_slippage_bps: int              # 5 default
    paper_fee_bps: int                   # 10 default (Binance taker)
    max_open_positions: int              # 20
    max_open_per_symbol: int             # 3
    db_path: Path
    data_dir: Path
    backups_dir: Path
    predictions_path: Path
    strategy_state_path: Path
    self_improve_log_path: Path
    edge_scan_history_path: Path
    anthropic_api_key: str | None
    binance_api_key: str | None
    binance_api_secret: str | None
    live_enabled: bool                   # must be flipped manually + Phase 9+

def load() -> GlobalConfig: ...
def time_now_ms() -> int: ...
def daily_budget() -> float: ...         # reads settings.json each call
```

### 3.2 `settings.json` (live-tunable)

```json
{
  "daily_budget_usd": 200.0,
  "active_strategies": ["nsigma_fade"],
  "kelly_fraction": 0.25,
  "live_enabled": false
}
```

Reloaded on every CLI invocation. The scheduler reloads every loop tick.

### 3.3 `.env` (secrets, gitignored)

```
ANTHROPIC_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=
MODE=PAPER
```

STATE: spec.

---

## 4. Persistence schema (`trades.db`)

SQLite, WAL mode, `PRAGMA synchronous=NORMAL`. Created by
`store.init_db()`. Schema versioning via a `schema_version` table; bump
on every change.

### 4.1 Tables

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,                   -- 'PAPER' or 'LIVE'
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    entry_ts INTEGER NOT NULL,            -- epoch ms
    side TEXT NOT NULL,                   -- 'LONG' or 'SHORT'
    entry_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    tp_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    horizon_bars INTEGER NOT NULL,
    timeout_ts INTEGER NOT NULL,          -- entry_ts + horizon_bars*tf_ms
    pred_p_up REAL,                       -- nullable; null for non-prob strategies
    edge REAL NOT NULL,
    estimator TEXT NOT NULL,              -- 'heuristic' | 'claude' | 'manual' | 'rule'
    regime TEXT,                          -- regime at entry
    reason TEXT NOT NULL,                 -- human-readable signal reason
    fee_bps_assumed INTEGER NOT NULL,
    slippage_bps_assumed INTEGER NOT NULL,

    status TEXT NOT NULL,                 -- 'OPEN' | 'WON' | 'LOST' | 'VOID' | 'TIMEOUT'
    exit_ts INTEGER,
    exit_price REAL,
    exit_reason TEXT,                     -- 'TP' | 'SL' | 'TIMEOUT' | 'MANUAL'
    pnl_usd REAL,
    fee_usd REAL,
    slippage_usd REAL
);

CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_strategy ON trades(strategy, status);
CREATE INDEX idx_trades_entry_ts ON trades(entry_ts);

CREATE TABLE bankroll (
    id INTEGER PRIMARY KEY,               -- 1 = paper aggregate, 2 = live, 100+ = per-strategy
    strategy TEXT,                        -- nullable for aggregate
    mode TEXT NOT NULL,
    balance REAL NOT NULL,
    initial_deposit REAL NOT NULL,
    peak_equity REAL NOT NULL,
    open_exposure REAL NOT NULL DEFAULT 0,
    drawdown_halted INTEGER NOT NULL DEFAULT 0,
    last_updated INTEGER NOT NULL
);

CREATE TABLE bankroll_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bankroll_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    delta REAL NOT NULL,
    balance_after REAL NOT NULL,
    note TEXT,
    trade_id INTEGER                      -- nullable; references trades.id when applicable
);

CREATE TABLE daily_snapshots (
    day TEXT PRIMARY KEY,                 -- 'YYYY-MM-DD' UTC
    paper_equity REAL NOT NULL,
    live_equity REAL,
    open_count INTEGER NOT NULL,
    trades_opened_today INTEGER NOT NULL,
    trades_closed_today INTEGER NOT NULL,
    pnl_today REAL NOT NULL,
    peak_equity REAL NOT NULL,
    drawdown_pct REAL NOT NULL
);

CREATE TABLE gate_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    gate TEXT NOT NULL,
    reason TEXT NOT NULL,
    size_usd REAL,
    pred_p_up REAL
);

CREATE INDEX idx_gate_failures_ts ON gate_failures(ts);
```

### 4.2 Invariants

- A trade row in `WON`/`LOST`/`TIMEOUT`/`VOID` has non-null `exit_ts`,
  `exit_price`, `exit_reason`, `pnl_usd`, `fee_usd`, `slippage_usd`.
- `bankroll.balance` after settle equals `initial_deposit + sum(pnl_usd
  for closed trades of that bankroll) - sum(fee_usd) - sum(slippage_usd)`
  within rounding (assert in a test).
- `bankroll_log.balance_after` is always equal to the row's balance at
  the moment of write (verified by replay test).

STATE: spec.

---

## 5. Exchange wrapper (`btcbot/exchange.py`)

```python
class Exchange:
    def __init__(self, name: str = "binance", sandbox: bool = False): ...

    def fetch_recent_candles(
        self, symbol: str, timeframe: str, n: int,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame: ...
    # columns: open_time(ms), open, high, low, close, volume
    # always sorted ascending; index = open_time

    def fetch_orderbook(
        self, symbol: str, depth: int = 20,
    ) -> dict: ...
    # {'bids': [(price, size), ...], 'asks': [...], 'ts': ms}

    def best_bid_ask(self, symbol: str) -> tuple[float, float]: ...

    def fillable_depth(
        self, symbol: str, side: Literal["buy","sell"],
        max_slippage_bps: int,
    ) -> float: ...
    # walks the L2 book; returns USD size fillable within budget

    def fetch_funding_rate(self, symbol: str) -> dict: ...   # Phase 8
    def fetch_funding_history(self, symbol: str, since_ms: int) -> pd.DataFrame: ...
```

Rate limiting: rely on ccxt's `enableRateLimit=True`. Wrap every call in a
small retry-with-backoff helper for 5xx and `ExchangeNotAvailable`.

`drop_unclosed=True` is the default and the codebase never overrides it
for signal-generation paths. The only legitimate `False` use is the
dashboard showing the current forming candle.

STATE: spec.

---

## 6. Data layer (`btcbot/data.py`)

### 6.1 Snapshot

```python
@dataclass(frozen=True)
class Snapshot:
    symbol: str
    timeframe: str
    ts: int                  # bar open_time, epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: dict[str, float]   # filled by indicators.add_indicators
    regime: str | None             # filled by indicators.classify_regime

    @property
    def id(self) -> str:
        # stable id used in predictions.jsonl and dedup
        return f"{self.symbol}|{self.timeframe}|{self.ts}"
```

### 6.2 BinanceVisionLoader

```python
class BinanceVisionLoader:
    BASE = "https://data.binance.vision/data/spot/monthly/klines"

    def ensure_history(
        self, symbol: str, timeframe: str,
        start: date, end: date,
    ) -> Path: ...
    # downloads any missing monthly ZIPs, returns path to consolidated parquet

    def load(
        self, symbol: str, timeframe: str,
        start: int | None = None, end: int | None = None,
    ) -> pd.DataFrame: ...
    # reads parquet, returns sorted dedup'd frame

    def detect_gaps(self, df: pd.DataFrame, tf_ms: int) -> list[tuple[int,int]]: ...
```

### 6.3 Replay iterator

```python
def replay(
    df: pd.DataFrame, indicator_spec: dict,
    warmup_bars: int = 300,
) -> Iterator[Snapshot]:
    # adds indicators + regime in-place once; yields Snapshot per row
    # after warmup_bars (so early bars with NaN indicators are skipped)
```

STATE: spec.

---

## 7. Indicators + regime (`btcbot/indicators.py`)

### 7.1 Indicator functions

All take a pandas Series or DataFrame, return a Series. No mutation.

```
ema(s: pd.Series, n: int) -> pd.Series
sma(s, n)
rsi(s, n=14)
atr(df, n=14)                     # uses high/low/close
bollinger(s, n=20, k=2.0) -> (mid, upper, lower)
zscore(s, n=20)
realized_vol(s, n=288)            # annualized vol estimate for 5m bars over 1d
donchian(df, n=20) -> (high, low)
vwap(df)                          # session VWAP — note session = UTC day
volume_zscore(df, n=288)
```

### 7.2 `add_indicators`

```python
def add_indicators(df: pd.DataFrame, spec: dict) -> pd.DataFrame: ...
# spec example:
# {
#   "ema_50": {"fn": "ema", "args": {"n": 50}},
#   "ema_200": {"fn": "ema", "args": {"n": 200}},
#   "rsi_14": {"fn": "rsi", "args": {"n": 14}},
#   "atr_14": {"fn": "atr", "args": {"n": 14}},
#   "z_20": {"fn": "zscore", "args": {"n": 20}},
# }
```

### 7.3 Regime classifier

```python
def classify_regime(df: pd.DataFrame) -> pd.Series:
    # requires columns: close, ema_50, ema_200, atr_14
    # returns Series of {trending_up, trending_down, ranging, high_vol, mixed}
```

Heuristic — not a model. Documented thresholds:

- `trending_up`: `close > ema_200` and `ema_50.diff(20) > 0` and
  `atr_pct < p75(atr_pct, 2000)`.
- `trending_down`: symmetric.
- `ranging`: `|close − ema_200| / close < 0.02` and
  `atr_pct < p50(atr_pct, 2000)`.
- `high_vol`: `atr_pct > p90(atr_pct, 2000)`.
- else `mixed`.

### 7.4 Leak test

Property test: for many random `n`, every indicator column at index n in
`f(df[:n+warmup])` equals the same column at index n in `f(df)`. Failure
means future data leaked.

STATE: spec.

---

## 8. Strategy framework

### 8.1 `Signal`

```python
@dataclass(frozen=True)
class Signal:
    snapshot: Snapshot
    strategy: str
    side: Literal["LONG", "SHORT"]
    entry_price: float          # the price we expect to pay (mid + half-spread)
    pred_p_up: float | None
    edge: float                 # after-cost edge in fractional terms
    size_usd: float
    tp_price: float
    sl_price: float
    horizon_bars: int
    reason: str
    estimator: str
```

### 8.2 ABC

```python
class Strategy(ABC):
    name: str
    required_indicators: dict

    @abstractmethod
    def evaluate(self, snapshot: Snapshot, cfg: StrategyConfig,
                 cost_bps: int) -> Signal | None: ...
```

### 8.3 Phase 3 strategy — `nsigma_fade`

```python
class NSigmaFade(Strategy):
    name = "nsigma_fade"
    required_indicators = {
        "z_20": {"fn": "zscore", "args": {"n": 20}},
        "atr_14": {"fn": "atr", "args": {"n": 14}},
        "ema_200": {"fn": "ema", "args": {"n": 200}},
    }

    def evaluate(self, snap, cfg, cost_bps):
        z = snap.indicators["z_20"]
        atr = snap.indicators["atr_14"]
        regime = snap.regime
        if regime not in {"ranging", "mixed"}:
            return None
        if z >= -2.0:
            return None                          # not stretched enough
        entry = snap.close
        sl = entry - 0.7 * atr
        tp = entry + 1.0 * atr
        # implicit p estimate from past calibration; Phase 3 starts at 0.55
        pred = 0.55
        b = (tp - entry) / (entry - sl)
        edge = (b*pred - (1-pred))/b - cost_bps/10_000
        if edge < cfg.min_edge:
            return None
        size = kelly_size(pred, b, cfg)
        return Signal(snap, self.name, "LONG", entry, pred, edge,
                      size, tp, sl, horizon_bars=12,
                      reason=f"z={z:.2f}, atr={atr:.2f}",
                      estimator="rule")
```

### 8.4 Phase 6 additions

- `breakout_donchian`: long on first close above `donchian_high(20)`,
  sl=`atr_14`, tp=`2*atr_14`, gated to `trending_up`.
- `momentum_ema_cross`: long after `ema_50 > ema_200` and a pullback
  touching `ema_50` from above. Sl below the touch low, tp at
  `+3*atr_14`.

### 8.5 Phase 5 `claude_pred`

Reads `predictions.jsonl` for `snap.id`. If present, uses
`pred_p_up` from that file. Otherwise returns `None` (no fallback).

STATE: spec.

---

## 9. Sizing model

### 9.1 Kelly

```python
def kelly_size(p: float, b: float, cfg: StrategyConfig) -> float:
    if b <= 0 or not (0 < p < 1):
        return 0.0
    q = 1 - p
    f = (b*p - q) / b
    f = max(0.0, f) * cfg.kelly_fraction
    stake = f * cfg.bankroll_usd
    return round(min(stake, cfg.max_position_usd), 2)
```

### 9.2 Cost model

```python
def cost_bps(
    symbol: str, side: str, size_usd: float,
    exchange: Exchange, cfg: GlobalConfig,
) -> int:
    fee = cfg.paper_fee_bps                          # round-trip baked in elsewhere
    bid, ask = exchange.best_bid_ask(symbol)
    half_spread = int(10_000 * (ask - bid) / ((ask + bid) / 2) / 2)
    depth = exchange.fillable_depth(symbol, "buy" if side == "LONG" else "sell",
                                    cfg.paper_slippage_bps)
    slip = cfg.paper_slippage_bps if depth >= size_usd else cfg.paper_slippage_bps * 3
    return fee + half_spread + slip
```

Round-trip means we charge `2 * (fee + slippage)` over the life of a trade
— half at entry, half at exit — but in `edge` we compute the full
round-trip cost upfront so the strategy gates correctly.

STATE: spec.

---

## 10. Engine / gates (`btcbot/engine.py`)

### 10.1 GateResult

```python
@dataclass(frozen=True)
class GateResult:
    ok: bool
    gate: str = ""
    reason: str = ""

PASS = GateResult(True)
```

### 10.2 Gates (each a pure predicate)

```python
def gate_already_open(symbol: str, entry_bar_ts: int, strategy: str) -> GateResult
def gate_position_cap(cfg) -> GateResult
def gate_per_symbol_cap(symbol: str, cfg) -> GateResult
def gate_daily_spend(strategy: str, stake: float) -> GateResult
def gate_exposure(strategy: str, stake: float) -> GateResult
def gate_drawdown_halt(strategy: str) -> GateResult
def gate_fillable_depth(signal: Signal, exchange: Exchange) -> GateResult
def gate_affordable(strategy: str, stake: float) -> GateResult
def gate_min_edge(signal: Signal, strategy_cfg) -> GateResult
def gate_min_confidence(signal: Signal, strategy_cfg) -> GateResult
def gate_active_strategy(strategy: str) -> GateResult       # reads settings.json
```

### 10.3 run_gates

```python
def run_gates(signal: Signal, exchange: Exchange,
              gcfg: GlobalConfig, scfg: StrategyConfig) -> GateResult:
    for g in (
        gate_active_strategy(signal.strategy),
        gate_drawdown_halt(signal.strategy),
        gate_min_edge(signal, scfg),
        gate_min_confidence(signal, scfg),
        gate_already_open(signal.snapshot.symbol, signal.snapshot.ts, signal.strategy),
        gate_position_cap(gcfg),
        gate_per_symbol_cap(signal.snapshot.symbol, gcfg),
        gate_daily_spend(signal.strategy, signal.size_usd),
        gate_exposure(signal.strategy, signal.size_usd),
        gate_fillable_depth(signal, exchange),
        gate_affordable(signal.strategy, signal.size_usd),
    ):
        if not g.ok:
            return g
    return PASS
```

Order matters: cheap pure checks first, exchange-touching checks last.

### 10.4 should_attempt (paper fill simulator)

```python
def should_attempt(signal: Signal, mode: str, now_ts: int) -> bool:
    if mode == "LIVE":
        return True
    # Paper: 95% of signals fill at the simulated price. Variability via
    # stable hash so the same (snapshot.id, day) always decides the same.
    h = hashlib.sha256(f"{signal.snapshot.id}|{now_ts//86_400_000}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < 0.95
```

STATE: spec.

---

## 11. Executor (`btcbot/executor.py`)

```python
class Executor:
    def __init__(self, exchange: Exchange, gcfg: GlobalConfig): ...

    def execute(self, signal: Signal) -> dict:
        if self.gcfg.mode == "PAPER":
            return self._execute_paper(signal)
        if self.gcfg.mode == "LIVE":
            return self._execute_live(signal)
        raise ConfigError(f"unknown mode {self.gcfg.mode}")

    def _execute_paper(self, signal) -> dict:
        # fill_price = signal.entry_price * (1 + slip/10000) for LONG buys
        # records trade row (status='OPEN'), deducts bankroll
        ...

    def _execute_live(self, signal) -> dict:
        raise NotImplementedError("live mode wired in Phase 9")
```

Returned dict:

```python
{
    "trade_id": int,
    "mode": "PAPER",
    "status": "OPEN",
    "fill_price": float,
    "fill_size_usd": float,
    "fee_bps_assumed": int,
    "slippage_bps_assumed": int,
    "ts": int,
}
```

STATE: spec.

---

## 12. Resolver (`btcbot/resolver.py`)

```python
def resolve_all_open(now_ts: int, exchange: Exchange, gcfg: GlobalConfig) -> list[dict]:
    """For every OPEN trade, check whether TP/SL/TIMEOUT fired since entry.
    Atomically settle. Idempotent — calling twice at the same now_ts is
    a no-op on already-closed trades."""
```

Per-trade procedure:

1. Fetch closed candles from `trade.entry_ts + tf_ms` to `min(now_ts,
   trade.timeout_ts)` for `trade.symbol/trade.timeframe`.
2. Iterate bar-by-bar in order. For each bar:
   - If `side == LONG`: did `low <= sl_price`? did `high >= tp_price`?
   - If `side == SHORT`: symmetric.
   - If both within range in the same bar, **conservatively settle SL
     first**. Document this once and use it everywhere (backtest, paper,
     live).
3. If no bar triggered and `now_ts >= trade.timeout_ts`: TIMEOUT at the
   timeout bar's close.
4. Call `store.settle_and_credit(trade_id, exit_price, exit_reason,
   fee_usd, slippage_usd)`.

Disambiguation: paper and backtest must agree byte-for-byte on this
resolution logic. Share a single `_resolve_one(trade, candles)` pure
function used by both.

STATE: spec.

---

## 13. Bankroll (`btcbot/bankroll.py`)

```python
def init_bankroll(strategy: str | None, mode: str,
                  initial_deposit: float) -> int: ...
def get(strategy: str | None, mode: str = "PAPER") -> dict: ...
def summary(strategy: str | None = None, mode: str = "PAPER") -> dict: ...
def deduct_stake(strategy: str | None, stake: float, note: str,
                 trade_id: int | None = None) -> float: ...
def credit_payout(strategy: str | None, payout: float, note: str,
                  trade_id: int | None = None) -> float: ...
def update_peak(strategy: str | None) -> None: ...
def drawdown_halted(strategy: str | None) -> bool: ...
def exposure_ok(strategy: str | None, new_stake: float,
                gcfg: GlobalConfig) -> bool: ...
def can_afford(strategy: str | None, stake: float) -> bool: ...
def balance(strategy: str | None) -> float: ...
def peak_equity(strategy: str | None) -> float: ...
```

`summary()` returns:

```python
{
    "strategy": "...",
    "mode": "PAPER",
    "balance": 512.34,
    "initial_deposit": 500.0,
    "open_exposure": 87.20,
    "total_equity": 599.54,
    "profit": 99.54,
    "return_pct": 0.199,
    "peak_equity": 612.10,
    "drawdown_pct": 0.020,
    "drawdown_halted": False,
}
```

Atomic update rule: every credit/deduct happens inside the *same*
transaction that updated the trade row (in `store.settle_and_credit` or
`executor._execute_paper`). Bankroll mutations from a free-standing
context (e.g. manual top-up) are rare and behind a CLI command, not
library code.

STATE: spec.

---

## 14. Store (`btcbot/store.py`)

```python
def init_db() -> None: ...
def migrate() -> None: ...

def record_trade(signal: Signal, fill: dict) -> int: ...
def settle_and_credit(trade_id: int, exit_ts: int, exit_price: float,
                      exit_reason: str, fee_usd: float, slippage_usd: float,
                      ) -> tuple[float, float]:
    """Single transaction:
        - update trades row to closed status
        - compute pnl_usd
        - credit/deduct bankroll for the trade's strategy
        - append bankroll_log
       Returns (pnl_usd, new_balance)."""

def open_positions(strategy: str | None = None) -> list[dict]: ...
def already_open(symbol: str, entry_bar_ts: int, strategy: str) -> bool: ...
def open_position_count(strategy: str | None = None) -> int: ...
def open_count_for_symbol(symbol: str) -> int: ...
def staked_today(strategy: str | None) -> float: ...
def performance_summary(strategy: str | None = None) -> dict: ...
def save_daily_snapshot(day_utc: str) -> None: ...
def record_gate_failure(ts, strategy, symbol, gate, reason,
                        size_usd=None, pred_p_up=None) -> None: ...
def query_trades(filters: dict, limit: int = 500) -> list[dict]: ...
```

Every public function takes an optional `conn` parameter so tests can
inject a fixture connection.

STATE: spec.

---

## 15. Backtest harness (`btcbot/backtest.py`)

### 15.1 Walk-forward

```python
class WalkForward:
    def __init__(self, train_window: int, test_window: int, step: int): ...
    def windows(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]: ...
```

### 15.2 Simulate

```python
def simulate(
    strategy: Strategy, snapshots: Iterator[Snapshot],
    cfg: StrategyConfig, gcfg: GlobalConfig,
    bankroll_usd: float,
) -> BacktestResult: ...
```

Internal loop:

- Maintain a virtual `open_positions` list (no DB).
- Per snapshot: first resolve any open positions against this bar using
  the *same* `_resolve_one` as live; then evaluate strategy; if signal
  passes a *backtest-only* version of `run_gates` (skip exchange-touching
  gates, simulate fillable_depth from historical volume × a fudge), open
  a virtual position with the same slippage + fee model.
- Aggregate trades.

### 15.3 BacktestResult

```python
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
    trades: list[dict]
    verdict: str        # "edge_confirmed" | "inconclusive" | "no_edge"
```

`verdict`:

- `edge_confirmed`: `n_trades >= 1000 and win_rate_wilson_lower > break_even`.
- `no_edge`: `n_trades >= 1000 and win_rate_wilson_upper < break_even`.
- otherwise `inconclusive`.

`break_even = 1 / (1 + b)` where `b = avg_tp_ret / avg_sl_ret` after
costs.

STATE: spec.

---

## 16. Calibration (`btcbot/calibration.py`)

```python
def reliability_diagram(
    trades: list[dict], bucket_edges: list[float] = None,
    slice_by: list[str] | None = None,
) -> dict: ...
```

Returns per-bucket and per-slice:

```python
{
    "buckets": [
        {"lo": 0.50, "hi": 0.55, "n": 124, "predicted_mean": 0.524,
         "realized": 0.508, "wilson_lower": 0.420, "wilson_upper": 0.595},
        ...
    ],
    "ece": 0.034,             # expected calibration error
    "brier": 0.247,           # Brier score
    "verdict": "miscalibrated_high_buckets",
}
```

CLI: `python run.py calibration --strategy nsigma_fade --by regime --since 90d`.

STATE: spec.

---

## 17. Self-improvement ladder (`btcbot/self_improve.py`)

### 17.1 Cell key

```python
@dataclass(frozen=True)
class Cell:
    strategy: str
    regime: str
    indicator_band: str       # e.g. "z<-2", "z<-3"
    side: str
```

### 17.2 State

```python
@dataclass
class CellState:
    cell: Cell
    tier: Literal["trial","exploratory","confirmed","disabled"]
    stake_multiplier: float    # 0.25 / 0.5 / 1.0 / 0
    days_in_tier: int
    rolling_win_rate: float
    rolling_n: int
    rolling_wilson_lower: float
    last_changed_at: int
    history: list[dict]        # promote/demote audit
```

Stored in `strategy_state.json`, atomic-written.

### 17.3 run()

```python
def run(now_ts: int) -> dict:
    state = _load_state()
    for cell, cs in state.items():
        recent = _recent_trades(cell, n=30)
        wilson = _wilson_lower(recent.wins, recent.n)
        _promote_or_demote(cs, wilson, recent.n, now_ts)
    _save_state(state)
    return summary
```

Rules:

- `trial → exploratory` if `days_in_tier >= 5` and `wilson_lower > 0` for
  5 consecutive daily evaluations and `n >= 20`.
- `exploratory → confirmed` if same for 10 consecutive days and `n >= 60`.
- `confirmed → exploratory` if `wilson_lower < 0` over rolling 30 trades.
- `exploratory → trial` same rule.
- `trial → disabled` if `n >= 30` and `wilson_upper < break_even`.

Audit append to `self_improve_log.jsonl`:

```json
{"ts": 1719_..., "cell": {...}, "from": "trial", "to": "exploratory",
 "wilson_lower": 0.013, "n": 24, "reason": "5 consecutive WLB days"}
```

STATE: spec.

---

## 18. Predictions / LLM-in-loop (`btcbot/predictions.py`)

### 18.1 File format — `predictions.jsonl`

One JSON object per line, append-only:

```json
{"snapshot_id": "BTC/USDT|5m|1719000000000",
 "pred_p_up": 0.58, "confidence": "med",
 "regime": "ranging", "rationale": "z=-2.4, no news in last 2h",
 "estimator": "claude-opus-4-7", "ts": 1719000090000}
```

### 18.2 API

```python
def load_predictions() -> dict[str, dict]: ...     # snapshot_id -> latest row
def get_prediction(snapshot_id: str) -> dict | None: ...
def append_prediction(row: dict) -> None: ...
def export_snapshots_todo(n: int, since_ts: int) -> Path: ...
```

### 18.3 Round-trip

- `run.py export-snapshots --n 200 --interesting` writes
  `predictions_todo.jsonl` containing snapshots flagged as interesting
  (extreme z-score, regime transitions, etc.).
- Operator hands the file to a Claude session with a fixed prompt
  (template in `prompts/predict_direction.md`).
- Claude returns one JSON object per line into `predictions.jsonl`.
- `run.py trade` picks up new predictions automatically.

STATE: spec.

---

## 19. CLI (`run.py`)

Single dispatcher. Subcommands:

```
python run.py init
python run.py status
python run.py download --symbol BTC/USDT --timeframe 5m --since 2022-01-01
python run.py trade                    # one tick: evaluate + execute
python run.py resolve                  # one tick: close anything resolvable
python run.py loop --interval 300      # combined trade+resolve loop (use serve.py instead)
python run.py backtest --strategy nsigma_fade --start ... --end ...
python run.py calibration --strategy ... --by regime --since 90d
python run.py self-improve             # nightly ladder run
python run.py export-snapshots --n 200 --interesting
python run.py report                   # text report
python run.py report --by-strategy
python run.py history --since 7d
python run.py bankroll [--strategy ...]
python run.py snapshot-day             # write daily_snapshots row
python run.py backup                   # copy DB to backups/
python run.py reconcile-live           # Phase 9+
```

Each subcommand is a `cmd_<name>(args)` function in `run.py`. The
dispatcher is a small `argparse` setup.

STATE: spec.

---

## 20. Scheduler (`serve.py`)

```python
def main():
    cfg = config.load()
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(tick_trade,   "interval", minutes=5, id="trade")
    scheduler.add_job(tick_resolve, "interval", minutes=5, id="resolve",
                      next_run_time=now+30s)        # offset to avoid races
    scheduler.add_job(daily_jobs,   "cron", hour=0, minute=5, id="daily")
    scheduler.start()
```

`daily_jobs` runs: `snapshot-day`, `self-improve`, `backup`.

The scheduler is single-process. Trade-then-resolve order is enforced by
the 30s offset; both jobs are idempotent so concurrency would not corrupt
state, but the offset is belt-and-braces.

Windows-friendliness: APScheduler `BlockingScheduler` runs in the
foreground; we deploy as a Task Scheduler entry pointing at
`run_serve.bat`.

STATE: spec.

---

## 21. Dashboard (`dashboard.py`)

Flask app, port 5050.

Routes:

- `/` — equity curve (aggregate + per strategy), open positions count,
  drawdown bar, regime over last 7d.
- `/trades` — paginated, filterable.
- `/strategies` — per-strategy bankroll, ladder state, Wilson CI table.
- `/calibration` — reliability diagrams (per strategy, per regime).
- `/regime` — price overlay with regime bands.
- `/audit` — `self_improve_log.jsonl` tail.
- `/gates` — last 24h gate-failure histogram.

No write endpoints. The dashboard is strictly read-only. If we want a
"force close position" command, it's a CLI subcommand, not a button.

STATE: spec.

---

## 22. Notifications (`notify.py`, optional)

Phase 10. Pluggable backends: Telegram, email, none.

Trigger events:

- Drawdown halt fires.
- A cell auto-promotes to `confirmed`.
- A cell auto-disables.
- Daily summary at 00:05 UTC.
- Exchange error rate > 5% over last hour.

Templates in `notify_templates/`. Each event has a stable message
shape so the operator's grep-able log of past notifications stays clean.

STATE: spec.

---

## 23. Tests

Single `tests/` directory, mirrors package layout.

### 23.1 Coverage targets

- `btcbot/store.py`: 100% of public functions.
- `btcbot/bankroll.py`: 100%.
- `btcbot/engine.py`: 100% (each gate has a yes-case and no-case test).
- `btcbot/resolver.py`: 100%.
- Strategies: each evaluate() has at least 3 cases (signal/no-signal-too-weak/no-signal-wrong-regime).

### 23.2 Property tests (hypothesis)

- Indicator leak test (every indicator, many prefixes).
- Settle atomicity: simulate exception at every step of
  `settle_and_credit`, assert DB invariant `balance = initial + sum(pnl)`.
- Resolver idempotency: call twice with same now_ts, second call is a
  no-op.
- Walk-forward determinism: same seed + same data → byte-equal trades.

### 23.3 Integration tests

- End-to-end paper trade: synthetic candles → snapshot → strategy →
  gates → executor → trades.db. Then 12 more synthetic candles, resolver
  closes at TP. Bankroll balance equals expected.
- Crash-and-restart: simulate process kill between `record_trade` and
  next `resolve`. After restart, resolver picks up the OPEN trade.

### 23.4 Fixtures

- `tests/fixtures/btc_5m_2024_01.parquet` — one month of real BTC 5m
  data, committed to the repo for reproducibility.
- `tests/fixtures/synthetic_*` — programmatically generated candles
  exercising every gate.

STATE: spec.

---

## 24. Logging + audit

- Library code uses `logging` with module-named loggers.
- Top-level entry points (`run.py`, `serve.py`) configure a structured
  JSONL handler writing to `logs/YYYY-MM-DD.jsonl`.
- Every gate decision (pass or fail) for a signal that made it past
  cheap gates is logged. Gate failures also write to `gate_failures`
  table (queryable from the dashboard).
- Every executed trade logs entry + exit with the trade_id.
- Log rotation: 30 days of daily files retained, older are gzipped to
  `logs/archive/`.

STATE: spec.

---

## 25. Backup + recovery

- `python run.py backup` copies `trades.db` to
  `backups/trades-YYYYMMDD-HHMMSS.db`. Keeps the last 30.
- Daily scheduled job runs the backup at 00:10 UTC.
- Quarterly drill: operator restores from a backup into a tempdir, runs
  `python run.py report` against it, verifies numbers match expectation.
- `data/parquet/` is *not* in the backup rotation — it's regeneratable
  from Binance Vision in under an hour.
- `predictions.jsonl`, `self_improve_log.jsonl`, `edge_scan_history.jsonl`
  are append-only — daily zipped snapshot kept under
  `backups/audit-YYYYMMDD.zip`.

STATE: spec.

---

## 26. Performance budget

- 5y of 5m BTC candles backtest: < 5 minutes wall-clock on this machine
  for a single strategy.
- One `trade` tick (load latest 300 bars, indicators, evaluate, gates,
  execute) end-to-end: < 5 seconds.
- One `resolve` tick (close any resolvable positions, max 20 open):
  < 10 seconds.
- Daily ladder + snapshot + backup: < 60 seconds.
- Dashboard cold load: < 2 seconds for 90 days of data.

If a budget is exceeded by 2×, we profile and fix before moving on.

STATE: spec.

---

## 27. CI + automation

GitHub-friendly (later) but locally:

- Pre-commit hook: `ruff check` + `black --check` + `pytest -q tests/`.
- Pre-push: full test suite including property tests.
- A `make` equivalent in a `tasks.py` (invoke-style) with targets:
  `tasks.lint`, `tasks.test`, `tasks.backtest`, `tasks.report`.

STATE: spec.

---

## 28. Live-mode wiring (Phase 9, dormant)

Even though we won't enable it for months, the design is specified here
so the code can be written defensively:

- `executor._execute_live`:
  1. Re-check `gcfg.live_enabled` and `MODE=LIVE`. If either is off,
     `raise LiveDisabledError`.
  2. Re-check API keys present. If not, raise.
  3. Query account balance. If < `signal.size_usd * 2`, refuse.
  4. Place limit order at `signal.entry_price` with `timeInForce=IOC`.
  5. Poll up to `LIVE_FILL_TIMEOUT_MS` (default 3000ms).
  6. On partial fill, cancel remainder, settle with actual filled size.
  7. Record trade row with `mode='LIVE'`, separate bankroll id.
- Live keys require **trade + read only**. Withdrawal-enabled keys are
  refused at startup with a fail-closed check.
- `python run.py reconcile-live` diffs the exchange-side trade history
  for the last 7 days against `trades.db`. Mismatches → write to
  `reconcile_mismatches.jsonl` and surface in dashboard.

STATE: spec, dormant until Phase 9.

---

## 29. Operator runbook

Daily:

```
python run.py status                  # quick health check
python run.py history --since 1d      # what closed today
```

Weekly:

```
python run.py report --by-strategy
python run.py calibration --strategy <name> --since 30d
```

Monthly:

```
python run.py backup
# review dashboard's /strategies page for ladder movement
# spot-check 10 random trades and confirm reason/regime match the data
```

Quarterly:

- Recovery drill (restore from backup).
- Audit pass: read the last quarter's `self_improve_log.jsonl` and ask
  "did the ladder make sensible decisions?"
- Re-evaluate phase. PLAN.md's Phase 12 decision applies at month 12;
  intermediate quarterly reviews are checkpoints, not decisions.

When things go wrong:

- Drawdown halt fires → don't immediately reset. Read the last 30 trades
  for that strategy, identify what changed, fix or disable, then reset.
- Exchange error storm → bot will log + back off. If it lasts > 1h, stop
  the scheduler and investigate.
- DB corruption → restore from last backup, replay any missing trades
  manually from logs (logs are the secondary source of truth).

STATE: spec.

---

## 30. Decision report shape (Phase 12)

`REPORT.md` written at month 12. Sections:

1. **Headline numbers.** Per strategy and aggregate: total trades, win
   rate (with Wilson CI), net PnL, Sharpe, max drawdown. Compared to
   buy-and-hold BTC over the same period.
2. **Calibration.** Per strategy: ECE, Brier, top 3 most-miscalibrated
   buckets. Honest assessment of whether `pred_p_up` meant anything.
3. **Ladder history.** How many cells reached `confirmed`. How long did
   they survive? Did any get re-demoted?
4. **Cost reality check.** Did the actual paper fee + slippage match
   what backtest assumed? If the live exchange spread widened during
   volatility, did the bot's gates correctly skip?
5. **Failure modes encountered.** Every bug, every drift, every
   surprise. Written candidly.
6. **Decision.** One of:
   - **Cautious live.** Specific micro-notional plan, monitoring rules,
     escalation criteria.
   - **Iterate.** Specific list of changes for "year 2 plan."
   - **Wind down.** Honest acknowledgment of no edge, archive the
     infrastructure, salvage the code/data for future research.

The decision must be defensible from the data alone, without reference
to "we put a year into this." Sunk cost is not evidence.

STATE: spec (writeup happens at month 12).

---

## Appendix A — File creation order (suggested)

For the developer implementing Phase 0 → Phase 4 from scratch, this is
the order that minimizes "needed it before I wrote it":

```
1.  pyproject.toml / requirements.txt
2.  btcbot/__init__.py
3.  btcbot/errors.py
4.  btcbot/config.py
5.  btcbot/store.py            (schema only, no business logic yet)
6.  btcbot/bankroll.py
7.  btcbot/exchange.py
8.  btcbot/data.py             (Snapshot dataclass + Vision loader)
9.  btcbot/indicators.py
10. btcbot/strategies/base.py
11. btcbot/strategy.py         (Signal + kelly + cost_model)
12. btcbot/strategies/nsigma_fade.py
13. btcbot/backtest.py
14. tests/                     (Phase 1-3 tests)
15. btcbot/engine.py
16. btcbot/executor.py
17. btcbot/resolver.py
18. run.py                     (CLI dispatcher)
19. serve.py
20. dashboard.py
21. btcbot/calibration.py
22. btcbot/predictions.py
23. btcbot/strategies/claude_pred.py
24. btcbot/self_improve.py
```

Each file under 400 lines if possible. If a file pushes past 400 lines,
that's a signal it wants to split.

STATE: spec.

---

## Appendix B — Anti-patterns to avoid

Documented so they don't sneak back in:

- ❌ Calling `datetime.now()` directly in library code. Use `now_ts`
  parameter.
- ❌ Reading `settings.json` from inside a hot loop. Cache and reload
  per tick at the scheduler level.
- ❌ Storing PnL in a separate file from the trade row. One transaction,
  one truth.
- ❌ A "quick" hyperopt sweep on a strategy. Curve-fitting. Don't.
- ❌ Manually editing `strategy_state.json` to "help" the ladder.
- ❌ Disabling the drawdown halt because "this time is different."
- ❌ Reading the unclosed last candle in any signal-generation path.
- ❌ Using a dict where a dataclass would clarify shape.
- ❌ Catching `Exception` broadly. Catch what you can handle, raise
  typed errors otherwise.
- ❌ Hard-coding thresholds inside strategy `evaluate()`. They live in
  `StrategyConfig` or per-strategy params, always.

STATE: rule, applies forever.

---

End of DEV_PLAN.md.
