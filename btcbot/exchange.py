from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd

from . import config
from .errors import ExchangeError


class Exchange:
    """Thin ccxt wrapper. Read-only by default; trade endpoints used only in Phase 9."""

    def __init__(self, name: str = "binance", sandbox: bool = False):
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise ExchangeError("ccxt not installed; pip install ccxt") from exc
        ex_cls = getattr(ccxt, name, None)
        if ex_cls is None:
            raise ExchangeError(f"unknown exchange {name!r}")
        kw: dict[str, Any] = {"enableRateLimit": True, "timeout": 20_000}
        cfg = config.load()
        if cfg.binance_api_key and cfg.binance_api_secret:
            kw["apiKey"] = cfg.binance_api_key
            kw["secret"] = cfg.binance_api_secret
        self._ex = ex_cls(kw)
        if sandbox:
            self._ex.set_sandbox_mode(True)
        self.name = name

    def _retry(self, fn, *args, **kwargs):
        last = None
        for i in range(4):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last = exc
                time.sleep(0.5 * (2**i))
        raise ExchangeError(f"{fn.__name__} failed after retries: {last}")

    def fetch_recent_candles(
        self, symbol: str, timeframe: str, n: int, drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        n = max(2, int(n))
        limit = min(1000, n + (1 if drop_unclosed else 0))
        raw = self._retry(self._ex.fetch_ohlcv, symbol, timeframe, None, limit)
        if not raw:
            raise ExchangeError("empty ohlcv response")
        df = pd.DataFrame(
            raw, columns=["open_time", "open", "high", "low", "close", "volume"]
        )
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        if drop_unclosed and len(df) >= 2:
            df = df.iloc[:-1].reset_index(drop=True)
        return df.tail(n).reset_index(drop=True)

    def fetch_orderbook(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        raw = self._retry(self._ex.fetch_order_book, symbol, depth)
        return {
            "bids": [(float(p), float(s)) for p, s in raw.get("bids", [])[:depth]],
            "asks": [(float(p), float(s)) for p, s in raw.get("asks", [])[:depth]],
            "ts": int(raw.get("timestamp") or config.time_now_ms()),
        }

    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        ob = self.fetch_orderbook(symbol, depth=1)
        if not ob["bids"] or not ob["asks"]:
            raise ExchangeError("empty orderbook")
        return ob["bids"][0][0], ob["asks"][0][0]

    def fillable_depth(
        self, symbol: str, side: Literal["buy", "sell"], max_slippage_bps: int,
    ) -> float:
        ob = self.fetch_orderbook(symbol, depth=20)
        levels = ob["asks"] if side == "buy" else ob["bids"]
        if not levels:
            return 0.0
        ref = levels[0][0]
        budget = ref * (1 + max_slippage_bps / 10_000) if side == "buy" else ref * (1 - max_slippage_bps / 10_000)
        usd = 0.0
        for price, size in levels:
            if (side == "buy" and price > budget) or (side == "sell" and price < budget):
                break
            usd += price * size
        return float(usd)

    def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        try:
            r = self._retry(self._ex.fetch_funding_rate, symbol)
            return {
                "symbol": symbol,
                "rate": float(r.get("fundingRate") or 0.0),
                "next_ts": int(r.get("fundingTimestamp") or 0),
            }
        except Exception as exc:
            raise ExchangeError(f"funding rate not available for {symbol}: {exc}") from exc

    def fetch_funding_history(self, symbol: str, since_ms: int) -> pd.DataFrame:
        try:
            raw = self._retry(self._ex.fetch_funding_rate_history, symbol, since_ms, 1000)
        except Exception as exc:
            raise ExchangeError(f"funding history fetch failed: {exc}") from exc
        if not raw:
            return pd.DataFrame(columns=["ts", "rate"])
        return pd.DataFrame(
            [{"ts": int(r["timestamp"]), "rate": float(r["fundingRate"])} for r in raw]
        )
