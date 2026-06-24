"""Per-strategy / per-pattern / per-regime P&L attribution.

Answers the question: "After N days, which strategies and patterns are
actually making me money?" Uses the same Wilson lower-bound discipline
as the rest of the bot.

CLI: python run.py edge-report [--since 7d]
Dashboard: rendered as a markdown block, plus a JSON endpoint.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, store
from .backtest import wilson_interval


@dataclass
class Bucket:
    key: str
    n: int = 0
    wins: int = 0  # status == 'WON'
    losses: int = 0
    timeouts: int = 0
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    fee_paid: float = 0.0
    deployed: float = 0.0
    open_count: int = 0
    by_status: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def wilson_lower(self) -> float:
        wlb, _ = wilson_interval(self.wins, self.n)
        return wlb

    @property
    def wilson_upper(self) -> float:
        _, wub = wilson_interval(self.wins, self.n)
        return wub

    @property
    def avg_pnl(self) -> float:
        return self.net_pnl / self.n if self.n else 0.0

    @property
    def roi_pct(self) -> float:
        return self.net_pnl / self.deployed if self.deployed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "n": self.n, "wins": self.wins, "losses": self.losses,
            "timeouts": self.timeouts, "open": self.open_count,
            "net_pnl": round(self.net_pnl, 2), "gross_pnl": round(self.gross_pnl, 2),
            "fee_paid": round(self.fee_paid, 2),
            "deployed": round(self.deployed, 2),
            "win_rate": round(self.win_rate, 4),
            "wilson_lower": round(self.wilson_lower, 4),
            "wilson_upper": round(self.wilson_upper, 4),
            "avg_pnl": round(self.avg_pnl, 2),
            "roi_pct": round(self.roi_pct, 4),
        }


def _add_trade(b: Bucket, t: dict) -> None:
    b.n += 1
    b.deployed += float(t.get("size_usd") or 0)
    status = t.get("status") or "OPEN"
    b.by_status[status] = b.by_status.get(status, 0) + 1
    if status == "WON":
        b.wins += 1
    elif status == "LOST":
        b.losses += 1
    elif status == "TIMEOUT":
        b.timeouts += 1
    elif status == "OPEN":
        b.open_count += 1
        b.n -= 1  # don't count open in the resolved sample
        return
    pnl = t.get("pnl_usd")
    fee = t.get("fee_usd") or 0.0
    slip = t.get("slippage_usd") or 0.0
    if pnl is not None:
        b.net_pnl += float(pnl)
        b.gross_pnl += float(pnl) + float(fee) + float(slip)
        b.fee_paid += float(fee) + float(slip)


def _bucketize(trades: list[dict], key_fn) -> dict[str, Bucket]:
    out: dict[str, Bucket] = {}
    for t in trades:
        k = key_fn(t)
        if k is None:
            continue
        b = out.setdefault(k, Bucket(key=k))
        _add_trade(b, t)
    return out


def report(since_days: int = 30) -> dict[str, Any]:
    since_ts = config.time_now_ms() - since_days * 86_400_000
    trades = store.query_trades({"since_ts": since_ts}, limit=10_000)

    by_strategy = _bucketize(trades, lambda t: t.get("strategy"))
    by_regime = _bucketize(trades, lambda t: t.get("regime") or "unknown")
    by_side = _bucketize(trades, lambda t: t.get("side"))

    # Patterns trade under names like 'pattern::<name>'. Split them out.
    pattern_trades = [t for t in trades if (t.get("strategy") or "").startswith("pattern::")]
    by_pattern = _bucketize(pattern_trades, lambda t: (t.get("strategy") or "").split("::", 1)[-1])

    # Discovered variants trade under 'parent::variant'.
    variant_trades = [
        t for t in trades
        if "::" in (t.get("strategy") or "") and not (t.get("strategy") or "").startswith("pattern::")
    ]
    by_variant = _bucketize(variant_trades, lambda t: t.get("strategy"))

    # Daily P&L attribution
    by_day = _bucketize(trades, lambda t: _day_key(t.get("exit_ts") or t.get("entry_ts")))

    def _sorted(d: dict[str, Bucket]) -> list[dict]:
        return sorted([b.to_dict() for b in d.values()],
                      key=lambda r: (-r["net_pnl"], -r["n"]))

    total = Bucket(key="ALL")
    for t in trades:
        _add_trade(total, t)

    return {
        "as_of": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "since_days": since_days,
        "total": total.to_dict(),
        "by_strategy": _sorted(by_strategy),
        "by_pattern": _sorted(by_pattern),
        "by_variant": _sorted(by_variant),
        "by_regime": _sorted(by_regime),
        "by_side": _sorted(by_side),
        "by_day": _sorted(by_day),
        "winners": _select_winners(by_strategy, by_pattern, by_variant),
        "losers": _select_losers(by_strategy, by_pattern, by_variant),
    }


def _day_key(ts_ms: int | None) -> str:
    if not ts_ms:
        return "open"
    return _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _select_winners(*buckets: dict[str, Bucket]) -> list[dict]:
    """Top-5 things that are actually making money, by net_pnl AND positive Wilson_lower."""
    rows: list[dict] = []
    for d in buckets:
        for b in d.values():
            if b.n >= 5 and b.net_pnl > 0:
                rows.append(b.to_dict())
    rows.sort(key=lambda r: (-r["net_pnl"], -r["wilson_lower"]))
    return rows[:10]


def _select_losers(*buckets: dict[str, Bucket]) -> list[dict]:
    rows: list[dict] = []
    for d in buckets:
        for b in d.values():
            if b.n >= 5 and b.net_pnl < 0:
                rows.append(b.to_dict())
    rows.sort(key=lambda r: (r["net_pnl"], r["wilson_lower"]))
    return rows[:10]


def render_markdown(rep: dict[str, Any]) -> str:
    """Compact markdown for both the CLI and the dashboard."""
    lines: list[str] = []
    t = rep["total"]
    lines.append(f"# Edge report (last {rep['since_days']} days)")
    lines.append("")
    lines.append(f"As of: {rep['as_of']}")
    lines.append("")
    lines.append("## Total")
    lines.append(f"- Trades resolved: **{t['n']}** (open: {t['open']})")
    lines.append(f"- Win rate: **{t['win_rate']*100:.1f}%** "
                 f"(95% CI: {t['wilson_lower']*100:.1f}% to {t['wilson_upper']*100:.1f}%)")
    lines.append(f"- Net P&L: **${t['net_pnl']:+,.2f}**")
    lines.append(f"- Capital deployed: ${t['deployed']:,.2f}  -> ROI: **{t['roi_pct']*100:+.2f}%**")
    lines.append(f"- Fees+slippage paid: ${t['fee_paid']:,.2f}")
    lines.append("")
    if rep["winners"]:
        lines.append("## Winners (positive net P&L, ≥5 trades)")
        for r in rep["winners"]:
            lines.append(f"- **{r['key']}** — {r['n']} trades, "
                         f"WR {r['win_rate']*100:.1f}% "
                         f"(WLB {r['wilson_lower']*100:.1f}%), "
                         f"net ${r['net_pnl']:+,.2f}")
        lines.append("")
    if rep["losers"]:
        lines.append("## Losers (negative net P&L, ≥5 trades)")
        for r in rep["losers"]:
            lines.append(f"- **{r['key']}** — {r['n']} trades, "
                         f"WR {r['win_rate']*100:.1f}%, "
                         f"net ${r['net_pnl']:+,.2f}")
        lines.append("")
    lines.append("## By strategy")
    lines.append(_md_table(rep["by_strategy"]))
    if rep["by_pattern"]:
        lines.append("## By pattern")
        lines.append(_md_table(rep["by_pattern"]))
    if rep["by_variant"]:
        lines.append("## By variant")
        lines.append(_md_table(rep["by_variant"]))
    lines.append("## By regime")
    lines.append(_md_table(rep["by_regime"]))
    lines.append("## By side")
    lines.append(_md_table(rep["by_side"]))
    lines.append("## By day")
    lines.append(_md_table(rep["by_day"]))
    return "\n".join(lines)


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no data yet_\n"
    out = ["| key | n | WR | WLB | net P&L | ROI | deployed |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:30]:
        out.append(f"| {r['key']} | {r['n']} | {r['win_rate']*100:.1f}% | "
                   f"{r['wilson_lower']*100:.1f}% | "
                   f"${r['net_pnl']:+,.2f} | {r['roi_pct']*100:+.2f}% | "
                   f"${r['deployed']:,.2f} |")
    return "\n".join(out) + "\n"


def write_daily_snapshot() -> Path | None:
    """Write reports/report_dayN.md where N counts days since deposit_date."""
    settings = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
    dep = settings.get("deposit_date")
    if not dep:
        return None
    today = _dt.datetime.now(_dt.timezone.utc).date()
    dep_date = _dt.date.fromisoformat(dep)
    day_n = (today - dep_date).days
    if day_n <= 0:
        return None
    rep = report(since_days=max(day_n, 1))
    out_dir = config.ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"report_day_{day_n:02d}.md"
    p.write_text(render_markdown(rep), encoding="utf-8")

    # also write a "latest" symlink-style copy
    latest = out_dir / "latest.md"
    latest.write_text(render_markdown(rep), encoding="utf-8")
    latest_json = out_dir / "latest.json"
    latest_json.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return p
