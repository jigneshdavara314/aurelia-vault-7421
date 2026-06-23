# AITradeCenter

A Bitcoin **paper-trading** bot, modeled on the `polymarket-ai` architecture.
Strategy framework + walk-forward backtest + paper execution loop +
calibration + self-improvement ladder + dashboard. Live execution path is
scaffolded but disabled.

See `PLAN.md` for the 12-month strategic roadmap and `DEV_PLAN.md` for the
implementation manual.

## Quickstart (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py init
python run.py status
```

## Backtest

```
python run.py download --symbol BTC/USDT --timeframe 5m --since 2023-01-01
python run.py backtest --strategy nsigma_fade --start 2024-01-01 --end 2024-12-31
```

## Paper-trade loop (local)

```
python serve.py
```

Or one-shot ticks:

```
python run.py trade
python run.py resolve
```

## Dashboard

```
python dashboard.py           # Flask on http://127.0.0.1:5050
python dashboard.py --write   # writes static dashboard.html
```

## Running free in the cloud (GitHub Actions)

The workflow in `.github/workflows/bot.yml` runs the bot hourly, commits
the updated `trades.db` back to the repo (so history persists), and
publishes the dashboard to GitHub Pages.

**One-time setup after pushing to GitHub:**

1. Repo **Settings → Actions → General → Workflow permissions** →
   select **Read and write permissions** (lets the bot commit the DB
   back).
2. Repo **Settings → Pages → Build and deployment → Source** →
   **GitHub Actions** (publishes the dashboard at
   `https://<you>.github.io/<repo>/`).
3. The bot runs automatically. Trigger a first run from the **Actions**
   tab (**Run workflow**) if you don't want to wait for the schedule.

No secrets are required — it runs in PAPER mode with no API keys.

## Mode safety

- `MODE=PAPER` (default). All execution is simulated.
- `MODE=LIVE` plus `live_enabled=true` in `settings.json` plus
  `BINANCE_API_KEY` / `BINANCE_API_SECRET` would route through
  `executor._execute_live` — currently raises `LiveDisabledError`.
  Per `PLAN.md` Phase 9, that flips only after the Phase 12 decision.

No promises about profit. The bot is built to **measure** whether
ideas have edge, not to print money. See `PLAN.md` §0 Operating
principles.
