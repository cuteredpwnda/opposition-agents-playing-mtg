"""Tests for ``src/utils/seeding.py`` and the ablation script entry point."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import phase_rs_rollout_sweep as sweep
from src.integrations.phase_rs import kg_enrichment_adapter
from src.integrations.phase_rs.runner import GameRunResult
from src.utils.seeding import derive_seed, set_global_seed


def test_set_global_seed_makes_random_reproducible():
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_derive_seed_is_deterministic():
    s1 = derive_seed(42, "wm", "small", 0)
    s2 = derive_seed(42, "wm", "small", 0)
    s3 = derive_seed(42, "wm", "small", 1)
    assert s1 == s2
    assert s1 != s3


def test_ablation_sweep_help_exits_zero():
    """The CLI parser at least produces help without crashing."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "phase_rs_rollout_sweep.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--pickers" in result.stdout


def test_sweep_records_incomplete_games_and_retry_attempts(tmp_path, monkeypatch):
    def result(winner, reason, trace):
        return GameRunResult(
            winner_seat=winner, reason=reason, our_seat=0, turns_observed=2,
            actions_sent=3, final_state=None, trace=trace,
        )

    timeout = result(None, "stream_timeout", [{"event": "decision"}])
    results = iter([
        timeout, result(0, "game_rules", [{"event": "game_over"}]),
        timeout, timeout,
        result(None, "game_rules", [{"event": "game_over"}]),
    ])
    monkeypatch.setattr(sweep, "_build_picker", lambda *args: object())
    monkeypatch.setattr(sweep, "run_game_sync", lambda **kwargs: next(results))

    async def forbidden_enrichment(**kwargs):
        raise AssertionError("Incomplete sweeps must not enrich the graph")

    monkeypatch.setattr(kg_enrichment_adapter, "enrich_from_phase_rs_traces", forbidden_enrichment)
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--pickers", "random", "--difficulties", "VeryEasy",
        "--ai-decks", "Red Deck Wins", "--games-per-cell", "3",
        "--max-retries", "2", "--output-dir", str(tmp_path), "--kg-enrich",
    ])
    assert sweep.run(sweep.parse_args()) == 1
    output = next(tmp_path.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    cell = summary["cells"][0]
    assert cell["completed_games"] == 2
    assert cell["incomplete_games"] == 1
    assert cell["draws"] == 1
    assert cell["win_rate"] == pytest.approx(0.5)
    assert cell["completion_rate"] == pytest.approx(2 / 3)
    rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert rows[0]["attempt_reasons"] == ["stream_timeout", "game_rules"]
    assert rows[1]["result"] == "incomplete"
    assert rows[1]["ai_deck"] == "Red Deck Wins"
    assert len(list((output / "traces").glob("*_attempt_01.jsonl"))) == 2
    progress = json.loads((output / "progress.json").read_text())
    assert progress["attempted_games"] == 3
    assert progress["completed_games"] == 2
    assert not (output / "kg_enrichment_report.json").exists()


def test_sweep_without_terminal_games_has_no_win_rate(tmp_path, monkeypatch):
    failure = GameRunResult(
        winner_seat=None, reason="picker_invalid_response", our_seat=0,
        turns_observed=0, actions_sent=0, final_state=None,
    )
    monkeypatch.setattr(sweep, "_build_picker", lambda *args: object())
    monkeypatch.setattr(sweep, "run_game_sync", lambda **kwargs: failure)
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--pickers", "random", "--difficulties", "VeryEasy",
        "--games-per-cell", "1", "--output-dir", str(tmp_path),
    ])
    assert sweep.run(sweep.parse_args()) == 1
    cell = json.loads((next(tmp_path.iterdir()) / "summary.json").read_text())["cells"][0]
    assert cell["win_rate"] is None
    assert cell["completion_rate"] == 0
    assert cell["draws"] == 0


@pytest.mark.parametrize("option", ["--games-per-cell", "--max-retries"])
def test_sweep_rejects_nonpositive_counts(tmp_path, monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["sweep", option, "0", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="must"):
        sweep.run(sweep.parse_args())
    assert not list(tmp_path.iterdir())
