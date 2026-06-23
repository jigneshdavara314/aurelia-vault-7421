from __future__ import annotations

from btcbot.calibration import reliability_diagram, verdict_for


def _trades(n_per_bucket: dict):
    out = []
    for p, info in n_per_bucket.items():
        wins, losses = info
        for _ in range(wins):
            out.append({"pred_p_up": p, "exit_reason": "TP", "regime": "ranging"})
        for _ in range(losses):
            out.append({"pred_p_up": p, "exit_reason": "SL", "regime": "ranging"})
    return out


def test_well_calibrated():
    trades = _trades({
        0.51: (51, 49), 0.55: (55, 45), 0.6: (60, 40), 0.65: (65, 35),
    })
    diag = reliability_diagram(trades)
    assert diag["ece"] < 0.05
    assert verdict_for(diag) in {"well_calibrated", "mildly_miscalibrated"}


def test_miscalibrated():
    trades = _trades({
        0.6: (10, 90), 0.65: (5, 95),
    })
    diag = reliability_diagram(trades)
    assert diag["ece"] > 0.3
    assert verdict_for(diag) == "miscalibrated"


def test_slice_by_regime():
    trades = [
        {"pred_p_up": 0.6, "exit_reason": "TP", "regime": "ranging"},
        {"pred_p_up": 0.6, "exit_reason": "SL", "regime": "ranging"},
        {"pred_p_up": 0.7, "exit_reason": "TP", "regime": "trending_up"},
    ]
    diag = reliability_diagram(trades, slice_by="regime")
    assert "slices" in diag
    assert "ranging" in diag["slices"]
    assert "trending_up" in diag["slices"]
