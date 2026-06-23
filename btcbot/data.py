from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import config
from .errors import DataError


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    timeframe: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: dict[str, float] = field(default_factory=dict)
    regime: str | None = None

    @property
    def id(self) -> str:
        return f"{self.symbol}|{self.timeframe}|{self.ts}"


_BINANCE_TFS = {
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d",
}


class BinanceVisionLoader:
    """Bulk historical OHLCV from data.binance.vision (free, no API key)."""

    BASE = "https://data.binance.vision/data/spot/monthly/klines"

    def __init__(self, cache_root: Path | None = None):
        self.cache_root = (cache_root or (config.DATA_DIR / "binance_vision")).resolve()
        self.parquet_root = (config.DATA_DIR / "parquet").resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.parquet_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        return symbol.replace("/", "").upper()

    def _month_zip(self, sym: str, tf: str, y: int, m: int) -> Path:
        return self.cache_root / sym / tf / f"{sym}-{tf}-{y:04d}-{m:02d}.zip"

    def _month_url(self, sym: str, tf: str, y: int, m: int) -> str:
        return f"{self.BASE}/{sym}/{tf}/{sym}-{tf}-{y:04d}-{m:02d}.zip"

    def _months(self, start: date, end: date) -> Iterator[tuple[int, int]]:
        d = date(start.year, start.month, 1)
        last = date(end.year, end.month, 1)
        while d <= last:
            yield d.year, d.month
            if d.month == 12:
                d = date(d.year + 1, 1, 1)
            else:
                d = date(d.year, d.month + 1, 1)

    def _download_month(self, sym: str, tf: str, y: int, m: int) -> Path | None:
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise DataError("requests not installed") from exc
        path = self._month_zip(sym, tf, y, m)
        if path.exists() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = self._month_url(sym, tf, y, m)
        try:
            r = requests.get(url, timeout=60)
        except Exception as exc:
            raise DataError(f"network error fetching {url}: {exc}") from exc
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise DataError(f"HTTP {r.status_code} for {url}")
        path.write_bytes(r.content)
        return path

    @staticmethod
    def _parse_zip(path: Path) -> pd.DataFrame:
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
        ]
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not names:
                return pd.DataFrame(columns=cols[:6])
            with z.open(names[0]) as f:
                df = pd.read_csv(f, header=None, names=cols)
        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        return df.dropna().reset_index(drop=True)

    def ensure_history(
        self, symbol: str, timeframe: str, start: date, end: date | None = None,
    ) -> Path:
        if timeframe not in _BINANCE_TFS:
            raise DataError(f"unsupported timeframe {timeframe!r}")
        end = end or datetime.now(timezone.utc).date().replace(day=1) - timedelta(days=1)
        sym = self._symbol_key(symbol)
        frames: list[pd.DataFrame] = []
        for y, m in self._months(start, end):
            zp = self._download_month(sym, timeframe, y, m)
            if zp is None:
                continue
            try:
                df = self._parse_zip(zp)
            except Exception as exc:
                raise DataError(f"failed to parse {zp}: {exc}") from exc
            if not df.empty:
                frames.append(df)
        out = self.parquet_root / f"{sym}-{timeframe}.parquet"
        if frames:
            full = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
            full = full.reset_index(drop=True)
            full.to_parquet(out)
        elif not out.exists():
            raise DataError(f"no data fetched and no cached parquet at {out}")
        return out

    def load(
        self, symbol: str, timeframe: str,
        start: int | None = None, end: int | None = None,
    ) -> pd.DataFrame:
        sym = self._symbol_key(symbol)
        path = self.parquet_root / f"{sym}-{timeframe}.parquet"
        if not path.exists():
            raise DataError(f"no cached parquet at {path}; call ensure_history first")
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df["open_time"] >= int(start)]
        if end is not None:
            df = df[df["open_time"] < int(end)]
        return df.reset_index(drop=True)

    @staticmethod
    def detect_gaps(df: pd.DataFrame, tf_ms: int) -> list[tuple[int, int]]:
        if len(df) < 2:
            return []
        ot = df["open_time"].to_numpy()
        diffs = ot[1:] - ot[:-1]
        gaps: list[tuple[int, int]] = []
        for i, d in enumerate(diffs):
            if d != tf_ms:
                gaps.append((int(ot[i]), int(ot[i + 1])))
        return gaps


def df_to_snapshots(
    df: pd.DataFrame, symbol: str, timeframe: str, warmup_bars: int = 0,
) -> Iterator[Snapshot]:
    if warmup_bars:
        df = df.iloc[warmup_bars:]
    for row in df.itertuples(index=False):
        yield Snapshot(
            symbol=symbol, timeframe=timeframe,
            ts=int(row.open_time),
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume),
        )


def replay(
    df: pd.DataFrame, symbol: str, timeframe: str,
    indicator_spec: dict | None = None, warmup_bars: int = 300,
) -> Iterator[Snapshot]:
    from . import indicators as ind
    work = df.copy()
    if indicator_spec:
        work = ind.add_indicators(work, indicator_spec)
        work = ind.classify_regime(work)
    feature_cols = [c for c in work.columns if c not in {
        "open_time", "open", "high", "low", "close", "volume", "regime"
    }]
    for i, row in enumerate(work.itertuples(index=False)):
        if i < warmup_bars:
            continue
        feats = {col: float(getattr(row, col)) for col in feature_cols
                 if pd.notna(getattr(row, col))}
        regime = getattr(row, "regime", None) if "regime" in work.columns else None
        if regime is not None and not isinstance(regime, str):
            regime = None
        yield Snapshot(
            symbol=symbol, timeframe=timeframe, ts=int(row.open_time),
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume),
            indicators=feats, regime=regime,
        )
