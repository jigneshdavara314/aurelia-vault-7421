from __future__ import annotations

from . import bankroll, config, store
from .errors import ExchangeError, LiveDisabledError
from .strategy import Signal


class Executor:
    def __init__(self, exchange=None):
        self.exchange = exchange
        self.gcfg = config.load()

    def execute(self, signal: Signal) -> dict:
        if self.gcfg.mode == "PAPER":
            return self._execute_paper(signal)
        if self.gcfg.mode == "LIVE":
            return self._execute_live(signal)
        raise ExchangeError(f"unknown mode {self.gcfg.mode}")

    def _execute_paper(self, signal: Signal) -> dict:
        slip_bps = self.gcfg.paper_slippage_bps
        if signal.side == "LONG":
            fill_price = signal.entry_price * (1 + slip_bps / 10_000)
        else:
            fill_price = signal.entry_price * (1 - slip_bps / 10_000)
        ts = config.time_now_ms()
        tf_ms = config.timeframe_ms(signal.snapshot.timeframe)
        timeout_ts = signal.snapshot.ts + signal.horizon_bars * tf_ms
        fill = {
            "mode": "PAPER",
            "ts": signal.snapshot.ts,
            "fill_price": float(fill_price),
            "fill_size_usd": float(signal.size_usd),
            "fee_bps_assumed": self.gcfg.paper_fee_bps,
            "slippage_bps_assumed": self.gcfg.paper_slippage_bps,
        }
        sd = signal.to_dict(timeout_ts)
        with store.conn_ctx() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                trade_id = store.record_trade(sd, fill, conn=conn)
                bankroll.deduct_stake(
                    signal.strategy, signal.size_usd, f"open:{signal.strategy}",
                    trade_id=trade_id, conn=conn,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {
            "trade_id": trade_id, "mode": "PAPER", "status": "OPEN",
            "fill_price": fill["fill_price"], "fill_size_usd": fill["fill_size_usd"],
            "fee_bps_assumed": fill["fee_bps_assumed"],
            "slippage_bps_assumed": fill["slippage_bps_assumed"],
            "ts": fill["ts"], "now_ts": ts,
        }

    def _execute_live(self, signal: Signal) -> dict:
        if not config.is_live_enabled():
            raise LiveDisabledError("live disabled in settings.json (live_enabled=false)")
        if not (self.gcfg.binance_api_key and self.gcfg.binance_api_secret):
            raise LiveDisabledError("BINANCE_API_KEY/SECRET missing")
        if self.exchange is None:
            raise LiveDisabledError("no exchange instance provided")
        try:
            bid, ask = self.exchange.best_bid_ask(signal.snapshot.symbol)
        except Exception as exc:
            raise LiveDisabledError(f"orderbook check failed: {exc}") from exc
        ref = ask if signal.side == "LONG" else bid
        if abs(ref - signal.entry_price) / signal.entry_price > 0.005:
            raise LiveDisabledError("market moved >50bps since signal; refusing")
        raise LiveDisabledError(
            "live order placement not implemented; Phase 9 — flip live_enabled and "
            "manually verify reconciliation before enabling."
        )
