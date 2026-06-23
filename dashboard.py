from __future__ import annotations

import datetime
import html
import math
from pathlib import Path

from flask import Flask, jsonify, request

from btcbot import bankroll, calibration, config, self_improve, store

app = Flask(__name__)


def _money(v: float | None, plus: bool = True) -> str:
    if v is None:
        return "-"
    if abs(v) < 0.005:
        return "$0.00"
    if v < 0:
        return f"-${abs(v):,.2f}"
    sign = "+" if plus else ""
    return f"{sign}${v:,.2f}"


def _cls(v: float | None) -> str:
    if v is None:
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _equity_points(days: int = 60) -> list[tuple[str, float]]:
    with store.conn_ctx() as c:
        rows = c.execute(
            """
            SELECT day, paper_equity FROM daily_snapshots
            ORDER BY day DESC LIMIT ?
            """, (days,),
        ).fetchall()
    rows = list(reversed(rows))
    return [(r["day"], float(r["paper_equity"])) for r in rows]


def _equity_chart(points: list[tuple[str, float]], w: int = 860, h: int = 240, pad: int = 34) -> str:
    if len(points) < 2:
        return ('<div class="note">Equity chart populates once daily snapshots accrue. '
                'They are written by <code>python run.py snapshot-day</code> '
                'or automatically at 00:05 UTC by serve.py.</div>')
    days = [p[0] for p in points]
    vals = [p[1] for p in points]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1
    span = vmax - vmin
    n = len(vals)

    def x(i: int) -> float:
        return pad + (w - 2 * pad) * i / (n - 1)

    def y(v: float) -> float:
        return h - pad - (h - 2 * pad) * (v - vmin) / span

    line_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    area = (f"M {x(0):.1f},{h-pad:.1f} L "
            + " L ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
            + f" L {x(n-1):.1f},{h-pad:.1f} Z")
    grid = ""
    for k in range(5):
        gv = vmin + span * k / 4
        gy = y(gv)
        grid += (f'<line x1="{pad}" y1="{gy:.1f}" x2="{w-pad}" y2="{gy:.1f}" class="grid"/>'
                 f'<text x="6" y="{gy+4:.1f}" class="axlab">${gv:,.0f}</text>')
    xlabs = ""
    for i in (0, n // 2, n - 1):
        xlabs += (f'<text x="{x(i):.1f}" y="{h-8}" class="axlab" text-anchor="middle">'
                  f'{days[i][5:]}</text>')
    up = vals[-1] >= vals[0]
    color = "#3fb950" if up else "#f85149"
    return f"""
      <svg viewBox="0 0 {w} {h}" class="chart" preserveAspectRatio="xMidYMid meet">
        <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="{color}" stop-opacity="0.35"/>
          <stop offset="1" stop-color="{color}" stop-opacity="0"/>
        </linearGradient></defs>
        {grid}
        <path d="{area}" fill="url(#g)"/>
        <polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="2.5"/>
        {xlabs}
      </svg>
    """


def _donut(pct: float, label: str, w: int = 130) -> str:
    r, cx, cy, sw = 48, w / 2, w / 2, 12
    circ = 2 * math.pi * r
    filled = circ * (pct / 100)
    color = "#3fb950" if pct >= 55 else ("#d8a23b" if pct >= 50 else "#f85149")
    return f"""
      <svg viewBox="0 0 {w} {w}" class="donut">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#30363d" stroke-width="{sw}"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}"
          stroke-dasharray="{filled:.1f} {circ:.1f}" stroke-linecap="round"
          transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy-2}" text-anchor="middle" class="donut-pct">{pct:.0f}%</text>
        <text x="{cx}" y="{cy+16}" text-anchor="middle" class="donut-lab">{label}</text>
      </svg>
    """


def _strategy_bars(rows: list[dict]) -> str:
    bars: list[str] = []
    for r in rows:
        n = r["n_closed"]
        wr = r["win_rate"] * 100 if n else 0
        won = r["won"]
        lost = r["lost"]
        color = "#3fb950" if wr >= 55 else ("#d8a23b" if wr >= 50 else "#f85149")
        bars.append(f"""
          <div class="bar-row">
            <div class="bar-name">{html.escape(r['strategy'])}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{wr:.0f}%;background:{color}"></div></div>
            <div class="bar-val">{wr:.0f}% <span class="bar-sub">({won}–{lost})</span></div>
            <div class="bar-pnl {_cls(r['pnl'])}">{_money(r['pnl'])}</div>
          </div>""")
    if not bars:
        return '<div class="note">No settled trades yet. The bot is waiting for signals.</div>'
    return f'<div class="bars">{"".join(bars)}</div>'


def _open_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="note">No open positions.</div>'
    body = []
    for t in rows:
        opened = datetime.datetime.fromtimestamp(t["entry_ts"] / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        body.append(f"""
        <tr>
          <td>{t['id']}</td>
          <td>{html.escape(t['strategy'])}</td>
          <td class="{ 'pos' if t['side']=='LONG' else 'neg' }">{t['side']}</td>
          <td>{html.escape(t['symbol'])}</td>
          <td>${t['entry_price']:,.2f}</td>
          <td>${t['size_usd']:,.2f}</td>
          <td>${t['tp_price']:,.2f}</td>
          <td>${t['sl_price']:,.2f}</td>
          <td>{html.escape(t.get('regime') or '')}</td>
          <td>{opened}</td>
        </tr>""")
    return f"""
      <table class="t">
        <thead><tr><th>id</th><th>strategy</th><th>side</th><th>symbol</th>
          <th>entry</th><th>size</th><th>tp</th><th>sl</th><th>regime</th><th>opened</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    """


def _settled_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="note">No settled trades yet.</div>'
    body = []
    for t in rows:
        opened = datetime.datetime.fromtimestamp(t["entry_ts"] / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        closed = (datetime.datetime.fromtimestamp(t["exit_ts"] / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M") if t.get("exit_ts") else "-")
        pnl = t.get("pnl_usd")
        body.append(f"""
        <tr>
          <td>{t['id']}</td>
          <td>{html.escape(t['strategy'])}</td>
          <td class="{ 'pos' if t['side']=='LONG' else 'neg' }">{t['side']}</td>
          <td>${t['entry_price']:,.2f}</td>
          <td>${(t.get('exit_price') or 0):,.2f}</td>
          <td>{html.escape(t.get('exit_reason') or '-')}</td>
          <td class="{_cls(pnl)}">{_money(pnl)}</td>
          <td>{html.escape(t['status'])}</td>
          <td>{opened}</td>
          <td>{closed}</td>
        </tr>""")
    return f"""
      <table class="t">
        <thead><tr><th>id</th><th>strategy</th><th>side</th>
          <th>entry</th><th>exit</th><th>via</th><th>pnl</th><th>status</th>
          <th>opened</th><th>closed</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    """


def _ladder_table(state: dict) -> str:
    if not state:
        return '<div class="note">No cells evaluated yet. Run <code>python run.py self-improve</code> after some trades settle.</div>'
    body = []
    for k, cs in sorted(state.items()):
        body.append(f"""
        <tr>
          <td>{html.escape(k)}</td>
          <td><span class="pill tier-{html.escape(cs.tier)}">{html.escape(cs.tier)}</span></td>
          <td>{cs.days_in_tier}</td>
          <td>{cs.rolling_n}</td>
          <td>{cs.rolling_win_rate*100:.1f}%</td>
          <td>{cs.rolling_wilson_lower*100:.1f}%</td>
          <td>{cs.rolling_wilson_upper*100:.1f}%</td>
        </tr>""")
    return f"""
      <table class="t">
        <thead><tr><th>cell</th><th>tier</th><th>days</th><th>n</th>
        <th>WR</th><th>WLB</th><th>WUB</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    """


def _gate_failures_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="note">No gate failures recorded yet.</div>'
    body = []
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r["ts"] / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        body.append(f"""
        <tr>
          <td>{ts}</td>
          <td>{html.escape(r['strategy'])}</td>
          <td>{html.escape(r['symbol'])}</td>
          <td>{html.escape(r['gate'])}</td>
          <td>{html.escape(r['reason'])}</td>
        </tr>""")
    return f"""
      <table class="t">
        <thead><tr><th>when</th><th>strategy</th><th>symbol</th><th>gate</th><th>reason</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    """


CSS = """
:root{--bg:#0d1117;--card:#161b22;--card2:#181d24;--ink:#e6edf3;--mute:#8b949e;--brd:#30363d;--acc:#58a6ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:24px}
h1{margin:0 0 4px 0;font-size:20px}
h2{margin:24px 0 12px 0;font-size:15px;color:var(--mute);font-weight:600;letter-spacing:.05em;text-transform:uppercase}
.sub{color:var(--mute);font-size:12px;margin-bottom:24px}
.pos{color:#3fb950}.neg{color:#f85149}.accent{color:#58a6ff}
.card{background:var(--card);border:1px solid var(--brd);border-radius:8px;padding:18px;margin-bottom:16px}
.note{color:var(--mute);font-size:13px;padding:14px 0}
.invest-card{background:linear-gradient(180deg,#1a2230 0%,#161b22 100%);border:1px solid var(--brd);border-radius:10px;padding:18px;margin-bottom:14px}
.ic-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.ic-box{background:var(--card2);border:1px solid var(--brd);border-radius:8px;padding:14px}
.ic-box.highlight{border-color:#1f6feb}
.ic-lab{font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.ic-val{font-size:22px;font-weight:700}
.ic-sub{font-size:11px;color:var(--mute);margin-top:4px}
.ic-foot{font-size:12px;color:var(--mute);margin-top:14px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
.stat{background:var(--card);border:1px solid var(--brd);border-radius:8px;padding:12px;text-align:center}
.s-val{font-size:18px;font-weight:700}.s-lab{font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.grid-2{display:grid;grid-template-columns:2fr 1fr;gap:16px}
.chart{width:100%;height:auto;display:block}
.grid{stroke:#21262d;stroke-width:1}.axlab{fill:#6e7681;font-size:10px;font-family:inherit}
.donut{width:130px;height:130px}.donut-pct{fill:var(--ink);font-size:22px;font-weight:700;font-family:inherit}
.donut-lab{fill:var(--mute);font-size:10px;font-family:inherit;text-transform:uppercase;letter-spacing:.05em}
.donut-wrap{display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px}
.donut-n{color:var(--mute);font-size:12px}
.bars{display:flex;flex-direction:column;gap:8px}
.bar-row{display:grid;grid-template-columns:160px 1fr 90px 80px;align-items:center;gap:12px;font-size:13px}
.bar-name{color:var(--ink)}.bar-track{height:8px;background:#21262d;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}.bar-val{font-size:13px}.bar-sub{color:var(--mute);font-size:11px}
.bar-pnl{text-align:right;font-variant-numeric:tabular-nums}
table.t{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.t th{text-align:left;padding:8px 10px;border-bottom:1px solid var(--brd);color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em;font-size:10px}
table.t td{padding:8px 10px;border-bottom:1px solid #21262d}
.pill{padding:2px 8px;border-radius:999px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.tier-trial{background:#21262d;color:var(--mute)}
.tier-exploratory{background:#1f3a5f;color:#8ab4f8}
.tier-confirmed{background:#1d4d2f;color:#3fb950}
.tier-disabled{background:#5a1d1d;color:#f85149}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;background:#1d4d2f;color:#3fb950;margin-left:8px}
.badge.live{background:#5a1d1d;color:#f85149}
nav{margin-bottom:18px;display:flex;gap:8px;flex-wrap:wrap}
nav button{background:var(--card);border:1px solid var(--brd);color:var(--ink);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px}
nav button.active{background:#1f3a5f;border-color:#1f6feb;color:#cce0ff}
.tab{display:none}.tab.active{display:block}
"""


JS = """
function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('btn-'+name).classList.add('active');
  history.replaceState(null,'','#'+name);
}
window.addEventListener('DOMContentLoaded',function(){
  const h=(location.hash||'#overview').slice(1);
  showTab(['overview','open','settled','strategies','ladder','gates'].includes(h)?h:'overview');
  setTimeout(()=>location.reload(),60000);
});
"""


def _gather() -> dict:
    cfg = config.load(force=True)
    store.init_db()
    bankroll.init_bankroll(strategy=None, mode="PAPER")
    summary = bankroll.aggregate_summary(mode="PAPER") or {}
    open_pos = store.open_positions()
    from btcbot import strategies as strat_mod
    by_strat = []
    for name in strat_mod.names():
        bankroll.init_bankroll(strategy=name, mode="PAPER")
        perf = store.performance_summary(strategy=name)
        opens = store.open_position_count(strategy=name)
        bk = bankroll.summary(strategy=name, mode="PAPER")
        by_strat.append({
            "strategy": name,
            "n_closed": perf["n_closed"],
            "won": perf["won"], "lost": perf["lost"], "timeout": perf["timeout"],
            "win_rate": perf["win_rate"], "pnl": perf["pnl"],
            "open": opens, "balance": bk.get("balance", 0),
        })
    state = self_improve.all_states()
    settled = store.query_trades({"status": "WON"}, limit=200)
    settled += store.query_trades({"status": "LOST"}, limit=200)
    settled += store.query_trades({"status": "TIMEOUT"}, limit=200)
    settled.sort(key=lambda r: r.get("exit_ts") or 0, reverse=True)
    with store.conn_ctx() as c:
        gate_fails = [dict(r) for r in c.execute(
            "SELECT * FROM gate_failures ORDER BY ts DESC LIMIT 100"
        ).fetchall()]
    equity = _equity_points(60)
    return {
        "cfg": cfg, "summary": summary, "open_pos": open_pos,
        "by_strat": by_strat, "state": state, "settled": settled,
        "gate_fails": gate_fails, "equity": equity,
    }


def build_html() -> str:
    d = _gather()
    cfg = d["cfg"]
    s = d["summary"]
    deposit = s.get("initial_deposit", 500.0)
    free_balance = s.get("balance", 0)
    on_stake = s.get("open_exposure", 0)
    net_profit = s.get("profit", 0)
    net_equity = s.get("total_equity", 0)
    drawdown = s.get("drawdown_pct", 0) * 100
    halted = s.get("drawdown_halted", False)
    by_strat = d["by_strat"]
    total_won = sum(r["won"] for r in by_strat)
    total_lost = sum(r["lost"] for r in by_strat)
    total_timeout = sum(r["timeout"] for r in by_strat)
    resolved = total_won + total_lost
    wr = (total_won / resolved * 100) if resolved else 0.0
    daily_budget = config.daily_budget()
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pcls = _cls(net_profit)
    mode_badge = '<span class="badge">PAPER</span>' if cfg.mode == "PAPER" else '<span class="badge live">LIVE</span>'
    halt_badge = '<span class="badge live">DRAWDOWN HALTED</span>' if halted else ""

    headline = f"""
    <div class="invest-card">
      <div class="ic-row">
        <div class="ic-box">
          <div class="ic-lab">Invested</div>
          <div class="ic-val">${deposit:,.2f}</div>
          <div class="ic-sub">paper capital</div>
        </div>
        <div class="ic-box">
          <div class="ic-lab">Net profit / loss</div>
          <div class="ic-val {pcls}">{_money(net_profit)}</div>
          <div class="ic-sub">{(net_profit/deposit*100) if deposit else 0:+.2f}% on deposit</div>
        </div>
        <div class="ic-box">
          <div class="ic-lab">On stake</div>
          <div class="ic-val">${on_stake:,.2f}</div>
          <div class="ic-sub">in open positions</div>
        </div>
        <div class="ic-box highlight">
          <div class="ic-lab">Net balance</div>
          <div class="ic-val accent">${free_balance:,.2f}</div>
          <div class="ic-sub">free to deploy</div>
        </div>
      </div>
      <div class="ic-foot">
        Total account value <b>${net_equity:,.2f}</b>
        &nbsp;=&nbsp; ${free_balance:,.2f} free &nbsp;+&nbsp; ${on_stake:,.2f} on stake
        &nbsp;·&nbsp; symbol <b>{cfg.symbol}</b> &nbsp;·&nbsp; timeframe <b>{cfg.timeframe}</b>
        &nbsp;·&nbsp; daily budget <b>${daily_budget:,.2f}</b>
        &nbsp;·&nbsp; drawdown <b>{drawdown:.2f}%</b>
      </div>
    </div>
    """

    stats = f"""
    <div class="stats">
      <div class="stat"><div class="s-val pos">{total_won}</div><div class="s-lab">Won</div></div>
      <div class="stat"><div class="s-val neg">{total_lost}</div><div class="s-lab">Lost</div></div>
      <div class="stat"><div class="s-val">{total_timeout}</div><div class="s-lab">Timeout</div></div>
      <div class="stat"><div class="s-val">{wr:.0f}%</div><div class="s-lab">Win rate</div></div>
      <div class="stat"><div class="s-val {_cls(net_profit)}">{_money(net_profit)}</div><div class="s-lab">Profit</div></div>
      <div class="stat"><div class="s-val">{resolved}</div><div class="s-lab">Resolved</div></div>
    </div>
    """

    overview_panel = f"""
    <div class="card grid-2">
      <div>
        <h2>Equity curve (last 60 days)</h2>
        {_equity_chart(d['equity'])}
      </div>
      <div class="donut-wrap">
        <h2>Win rate</h2>
        {_donut(wr, f"{total_won}/{resolved or 1}")}
        <div class="donut-n">{resolved} resolved trades</div>
      </div>
    </div>
    <div class="card">
      <h2>Per-strategy performance</h2>
      {_strategy_bars(by_strat)}
    </div>
    """

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>btcbot dashboard</title>
<style>{CSS}</style>
</head><body>
<h1>btcbot {mode_badge} {halt_badge}</h1>
<div class="sub">Generated {generated} · auto-refresh 60s · active strategies: {', '.join(config.active_strategies())}</div>

<nav>
  <button id="btn-overview" onclick="showTab('overview')">Overview</button>
  <button id="btn-open" onclick="showTab('open')">Open ({len(d['open_pos'])})</button>
  <button id="btn-settled" onclick="showTab('settled')">Settled ({len(d['settled'])})</button>
  <button id="btn-strategies" onclick="showTab('strategies')">Strategies</button>
  <button id="btn-ladder" onclick="showTab('ladder')">Ladder</button>
  <button id="btn-gates" onclick="showTab('gates')">Gates</button>
</nav>

{headline}
{stats}

<div id="tab-overview" class="tab">{overview_panel}</div>

<div id="tab-open" class="tab"><div class="card"><h2>Open positions</h2>{_open_table(d['open_pos'])}</div></div>

<div id="tab-settled" class="tab"><div class="card"><h2>Settled trades</h2>{_settled_table(d['settled'])}</div></div>

<div id="tab-strategies" class="tab"><div class="card"><h2>Per-strategy detail</h2>{_strategy_bars(by_strat)}</div></div>

<div id="tab-ladder" class="tab"><div class="card"><h2>Self-improvement ladder</h2>{_ladder_table(d['state'])}</div></div>

<div id="tab-gates" class="tab"><div class="card"><h2>Recent gate failures (last 100)</h2>{_gate_failures_table(d['gate_fails'])}</div></div>

<script>{JS}</script>
</body></html>"""


@app.route("/")
def index():
    return build_html()


@app.route("/api/state.json")
def state_json():
    d = _gather()
    return jsonify({
        "mode": d["cfg"].mode, "symbol": d["cfg"].symbol, "timeframe": d["cfg"].timeframe,
        "summary": d["summary"], "open_count": len(d["open_pos"]),
        "settled_count": len(d["settled"]),
        "by_strategy": d["by_strat"],
        "open_positions": d["open_pos"],
    })


@app.route("/api/calibration/<strategy>")
def calibration_route(strategy: str):
    trades = store.query_trades({"strategy": strategy}, limit=5000)
    diag = calibration.reliability_diagram(trades, slice_by="regime")
    return jsonify({"strategy": strategy, "diagram": diag,
                    "verdict": calibration.verdict_for(diag)})


@app.route("/static/dashboard.html")
def static_html():
    return build_html()


def write_static(path: Path | None = None) -> Path:
    out = path or (config.ROOT / "dashboard.html")
    out.write_text(build_html(), encoding="utf-8")
    return out


def main(host: str = "127.0.0.1", port: int = 5050) -> None:
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="write dashboard.html and exit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    args = p.parse_args()
    if args.write:
        out = write_static()
        print(f"wrote {out}")
    else:
        main(args.host, args.port)
