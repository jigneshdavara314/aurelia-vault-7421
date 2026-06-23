from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BACKUPS_DIR = ROOT / "backups"
LOGS_DIR = ROOT / "logs"
DB_PATH = ROOT / "trades.db"
SETTINGS_PATH = ROOT / "settings.json"
PREDICTIONS_PATH = ROOT / "predictions.jsonl"
PREDICTIONS_TODO_PATH = ROOT / "predictions_todo.jsonl"
STRATEGY_STATE_PATH = ROOT / "strategy_state.json"
SELF_IMPROVE_LOG_PATH = ROOT / "self_improve_log.jsonl"
EDGE_SCAN_HISTORY_PATH = ROOT / "edge_scan_history.jsonl"

for d in (DATA_DIR, BACKUPS_DIR, LOGS_DIR, DATA_DIR / "binance_vision", DATA_DIR / "parquet"):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    min_edge: float = 0.02
    min_confidence: float = 0.52
    kelly_fraction: float = 0.25
    max_position_usd: float = 50.0
    min_liquidity_usd: float = 1000.0
    max_spread_bps: int = 20
    min_price: float = 0.0
    max_price: float = float("inf")
    bankroll_usd: float = 500.0
    regime_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_regime(self, regime: str | None) -> StrategyConfig:
        if not regime or regime not in self.regime_overrides:
            return self
        return replace(self, **self.regime_overrides[regime])


PROFILES: dict[str, StrategyConfig] = {
    "conservative": StrategyConfig(
        name="conservative",
        min_edge=0.03,
        min_confidence=0.55,
        kelly_fraction=0.20,
        max_position_usd=25.0,
    ),
    "moderate": StrategyConfig(
        name="moderate",
        min_edge=0.02,
        min_confidence=0.53,
        kelly_fraction=0.25,
        max_position_usd=50.0,
    ),
    "aggressive": StrategyConfig(
        name="aggressive",
        min_edge=0.015,
        min_confidence=0.52,
        kelly_fraction=0.35,
        max_position_usd=100.0,
    ),
}


@dataclass
class GlobalConfig:
    mode: str = "PAPER"
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    daily_budget_usd: float = 200.0
    drawdown_halt_frac: float = 0.20
    agg_exposure_frac: float = 0.50
    paper_slippage_bps: int = 5
    paper_fee_bps: int = 10
    max_open_positions: int = 20
    max_open_per_symbol: int = 3
    initial_deposit: float = 500.0
    profile: str = "moderate"
    active_strategies: list[str] = field(default_factory=lambda: ["nsigma_fade"])
    kelly_fraction: float = 0.25
    live_enabled: bool = False
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    backups_dir: Path = BACKUPS_DIR
    logs_dir: Path = LOGS_DIR
    predictions_path: Path = PREDICTIONS_PATH
    predictions_todo_path: Path = PREDICTIONS_TODO_PATH
    strategy_state_path: Path = STRATEGY_STATE_PATH
    self_improve_log_path: Path = SELF_IMPROVE_LOG_PATH
    edge_scan_history_path: Path = EDGE_SCAN_HISTORY_PATH
    anthropic_api_key: str | None = None
    binance_api_key: str | None = None
    binance_api_secret: str | None = None


_CONFIG: GlobalConfig | None = None


def _read_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"settings.json invalid: {exc}") from exc


def load(force: bool = False) -> GlobalConfig:
    global _CONFIG
    if _CONFIG is not None and not force:
        return _CONFIG
    s = _read_settings()
    cfg = GlobalConfig(
        mode=os.environ.get("MODE", "PAPER"),
        daily_budget_usd=float(s.get("daily_budget_usd", 200.0)),
        active_strategies=list(s.get("active_strategies", ["nsigma_fade"])),
        kelly_fraction=float(s.get("kelly_fraction", 0.25)),
        live_enabled=bool(s.get("live_enabled", False)),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        binance_api_key=os.environ.get("BINANCE_API_KEY") or None,
        binance_api_secret=os.environ.get("BINANCE_API_SECRET") or None,
    )
    _CONFIG = cfg
    return cfg


def daily_budget() -> float:
    return float(_read_settings().get("daily_budget_usd", 200.0))


def active_strategies() -> list[str]:
    return list(_read_settings().get("active_strategies", ["nsigma_fade"]))


def kelly_fraction() -> float:
    return float(_read_settings().get("kelly_fraction", 0.25))


def is_live_enabled() -> bool:
    return bool(_read_settings().get("live_enabled", False))


def time_now_ms() -> int:
    return int(time.time() * 1000)


def timeframe_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    if unit == "s":
        return n * 1_000
    raise ConfigError(f"unsupported timeframe {timeframe!r}")


def profile(name: str = "moderate") -> StrategyConfig:
    if name not in PROFILES:
        raise ConfigError(f"unknown profile {name!r}")
    return PROFILES[name]
