from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Each test gets its own trades.db / state files."""
    from btcbot import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(cfg_mod, "DB_PATH", tmp_path / "trades.db", raising=False)
    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data", raising=False)
    monkeypatch.setattr(cfg_mod, "BACKUPS_DIR", tmp_path / "backups", raising=False)
    monkeypatch.setattr(cfg_mod, "LOGS_DIR", tmp_path / "logs", raising=False)
    monkeypatch.setattr(cfg_mod, "SETTINGS_PATH", tmp_path / "settings.json", raising=False)
    monkeypatch.setattr(cfg_mod, "PREDICTIONS_PATH", tmp_path / "predictions.jsonl", raising=False)
    monkeypatch.setattr(cfg_mod, "PREDICTIONS_TODO_PATH", tmp_path / "predictions_todo.jsonl",
                        raising=False)
    monkeypatch.setattr(cfg_mod, "STRATEGY_STATE_PATH", tmp_path / "strategy_state.json",
                        raising=False)
    monkeypatch.setattr(cfg_mod, "SELF_IMPROVE_LOG_PATH", tmp_path / "self_improve_log.jsonl",
                        raising=False)
    monkeypatch.setattr(cfg_mod, "EDGE_SCAN_HISTORY_PATH", tmp_path / "edge_scan_history.jsonl",
                        raising=False)
    monkeypatch.setattr(cfg_mod, "_CONFIG", None, raising=False)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "backups").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "settings.json").write_text(
        '{"daily_budget_usd": 1000.0, "active_strategies": ["nsigma_fade","breakout_donchian",'
        '"momentum_ema_cross","claude_pred","funding_arb"], "kelly_fraction": 0.25, '
        '"live_enabled": false}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MODE", "PAPER")

    cfg_mod.load(force=True)
    yield
