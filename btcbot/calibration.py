from __future__ import annotations

from typing import Any

from .backtest import wilson_interval


DEFAULT_EDGES = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.70, 0.80, 0.95]


def _bucket(p: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i + 1]:
            return i
    return len(edges) - 2


def reliability_diagram(
    trades: list[dict], bucket_edges: list[float] | None = None,
    slice_by: str | None = None,
) -> dict[str, Any]:
    edges = bucket_edges or DEFAULT_EDGES
    closed = [t for t in trades if t.get("exit_reason") in {"TP", "SL", "TIMEOUT"}]

    def _aggregate(rows: list[dict]) -> dict:
        buckets: list[dict] = []
        for i in range(len(edges) - 1):
            buckets.append({
                "lo": edges[i], "hi": edges[i + 1],
                "n": 0, "wins": 0, "predicted_sum": 0.0,
            })
        for t in rows:
            p = t.get("pred_p_up")
            if p is None:
                continue
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            if not (0 < p < 1):
                continue
            idx = _bucket(p, edges)
            buckets[idx]["n"] += 1
            buckets[idx]["predicted_sum"] += p
            if t.get("exit_reason") == "TP":
                buckets[idx]["wins"] += 1
        out = []
        n_total = sum(b["n"] for b in buckets)
        ece = 0.0
        brier_sum = 0.0
        brier_n = 0
        for b in buckets:
            if b["n"] == 0:
                continue
            predicted_mean = b["predicted_sum"] / b["n"]
            realized = b["wins"] / b["n"]
            lo, hi = wilson_interval(b["wins"], b["n"])
            ece += (b["n"] / n_total if n_total else 0) * abs(predicted_mean - realized)
            out.append({
                "lo": b["lo"], "hi": b["hi"], "n": b["n"],
                "predicted_mean": round(predicted_mean, 4),
                "realized": round(realized, 4),
                "wilson_lower": round(lo, 4), "wilson_upper": round(hi, 4),
            })
        for t in rows:
            p = t.get("pred_p_up")
            if p is None:
                continue
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            y = 1 if t.get("exit_reason") == "TP" else 0
            brier_sum += (p - y) ** 2
            brier_n += 1
        brier = brier_sum / brier_n if brier_n else 0.0
        return {"buckets": out, "ece": round(ece, 4), "brier": round(brier, 4), "n_total": n_total}

    if slice_by:
        slices: dict[str, list[dict]] = {}
        for t in closed:
            key = str(t.get(slice_by) or "unknown")
            slices.setdefault(key, []).append(t)
        return {
            "overall": _aggregate(closed),
            "slices": {k: _aggregate(v) for k, v in slices.items()},
            "slice_by": slice_by,
        }
    return _aggregate(closed)


def verdict_for(diagram: dict) -> str:
    overall = diagram if "buckets" in diagram else diagram.get("overall", {})
    ece = overall.get("ece", 0.0)
    if ece < 0.03:
        return "well_calibrated"
    if ece < 0.07:
        return "mildly_miscalibrated"
    return "miscalibrated"
