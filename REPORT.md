# btcbot — Month 12 Decision Report (TEMPLATE)

This file is filled in at month 12 (end of `PLAN.md` Phase 12) using data
collected over 12 months of paper trading. Until then, the sections below
are placeholders.

## 1. Headline numbers

| metric | aggregate | nsigma_fade | breakout_donchian | momentum_ema_cross | claude_pred |
|---|---|---|---|---|---|
| trades | – | – | – | – | – |
| win rate | – | – | – | – | – |
| Wilson 95% CI | – | – | – | – | – |
| net P&L | – | – | – | – | – |
| Sharpe | – | – | – | – | – |
| max DD | – | – | – | – | – |
| vs buy-and-hold | – | – | – | – | – |

## 2. Calibration

Per strategy ECE, Brier score, top 3 most-miscalibrated buckets, per-regime
slice. Compare heuristic vs claude_pred.

## 3. Ladder history

How many cells reached `confirmed`. Average days at each tier. Cells that
were promoted then demoted.

## 4. Cost reality check

Did real paper slippage + fee match what backtest assumed? Where did the
gap show up?

## 5. Failure modes encountered

Candid log of every bug, every regime that broke a strategy, every
surprise. Sunk-cost-blind.

## 6. Decision

One of:

- **Cautious live.** Specify micro-notional plan, monitoring rules,
  escalation criteria.
- **Iterate.** Specific list of changes for Year 2.
- **Wind down.** Honest acknowledgment of no edge, archive infrastructure.

The decision must be defensible from §1–§5 alone.
