from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_phase_rs_ablation as ablation
from src.integrations.phase_rs.runner import GameRunResult


def _result(winner: int | None = 0, reason: str = "game_rules") -> GameRunResult:
    return GameRunResult(
        winner_seat=winner, reason=reason, our_seat=0,
        turns_observed=4, actions_sent=12, final_state={},
        trace=[{"event": "game_over", "winner": winner}],
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_result(), "win"),
        (_result(1), "loss"),
        (_result(None), "draw"),
        (replace(_result(None, "stream_timeout"), trace=[]), "incomplete"),
        (replace(_result(None, "action_cap"), trace=[]), "incomplete"),
        (replace(_result(0, "action_rejected"), trace=[]), "incomplete"),
    ],
)
def test_outcome_requires_terminal_evidence(result, expected):
    assert ablation._outcome(result) == expected


def _args(tmp_path: Path, *extra: str):
    return ablation.parse_args([
        "--games", "2", "--picker", "kl_control",
        "--objectives", "kl", "efe_infogain", "efe_ambiguity",
        "--our-deck-file", "deck.txt", "--format", "Modern",
        "--output-dir", str(tmp_path), *extra,
    ])


def _mock_runtime(monkeypatch):
    monkeypatch.setattr(
        ablation, "load_deck_data",
        lambda _: {"main_deck": ["Mountain"] * 60},
    )
    monkeypatch.setattr(ablation, "_revision", lambda _: "test-revision")
    monkeypatch.setattr(
        ablation, "_make_picker", lambda args, seed: (args.objective, seed)
    )


@pytest.mark.parametrize("name", ["world_model", "fusion"])
def test_explicit_checkpoint_and_model_options_reach_factory(tmp_path, monkeypatch, name):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    args = ablation.parse_args([
        "--picker", f"agent:{name}", "--agent-checkpoint", str(checkpoint),
        "--agent-mode", "dream_search", "--agent-deterministic",
        "--dream-rollouts", "2", "--dream-depth", "3", "--ollama-model", "test-model",
    ])
    calls = []
    monkeypatch.setattr(ablation, "make_agent",
                        lambda *a, **kw: calls.append((a, kw)) or object())
    ablation._make_picker(args, seed=7)
    options = calls[0][1]
    assert options["checkpoint"] == str(checkpoint)
    assert options["dream_rollouts"] == 2 and options["dream_depth"] == 3
    if name == "world_model":
        assert options["mode"] == "dream_search" and options["deterministic"] is True
    else:
        assert options["llm_model"] == "test-model"


def test_missing_checkpoint_is_explicit_failure(tmp_path):
    args = ablation.parse_args([
        "--picker", "agent:world_model", "--agent-checkpoint", str(tmp_path / "missing.pt"),
    ])
    with pytest.raises(ValueError, match="Checkpoint does not exist"):
        ablation._make_picker(args, seed=7)


def test_objective_schedule_and_outputs(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    calls = []

    def run_game(**kwargs):
        calls.append(kwargs)
        return _result()

    monkeypatch.setattr(ablation, "run_game_sync", run_game)
    assert ablation.run(_args(tmp_path)) == 0
    assert [call["picker"] for call in calls] == [
        (objective, seed)
        for seed in (7, 8)
        for objective in ("kl", "efe_infogain", "efe_ambiguity")
    ]
    assert all(call["format_name"] == "Modern" for call in calls)
    output = next(tmp_path.iterdir())
    manifest = json.loads((output / "config.json").read_text())
    assert manifest["phase_rs_commit"] == "test-revision"
    assert len(manifest["deck_sha256"]) == 64
    assert "picker only" in manifest["seed_scope"]
    rows = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert [row["seed"] for row in rows] == [7, 7, 7, 8, 8, 8]
    assert all(row["attempts"] == 1 for row in rows)
    assert len(list((output / "traces").glob("game_*.jsonl"))) == 6
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_games"] == 6
    for cell in summary["by_objective"].values():
        assert cell["wins"] == 2
        assert cell["win_rate"] == 1.0
        assert cell["win_rate_ci95"] == pytest.approx([0.3423802275, 1.0])


def test_incomplete_games_are_not_draws_and_fail_run(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    results = iter([
        _result(),
        replace(_result(None, "stream_timeout"), trace=[]),
        replace(_result(None, "stream_timeout"), trace=[]),
        _result(1),
    ])
    monkeypatch.setattr(ablation, "run_game_sync", lambda **_: next(results))
    assert ablation.run(_args(tmp_path, "--games", "1")) == 1
    output = next(tmp_path.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["games"] == 3
    assert summary["completed_games"] == 2
    assert summary["draws"] == 0
    assert summary["incomplete_games"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["by_objective"]["efe_infogain"]["win_rate"] is None
    assert summary["by_objective"]["efe_infogain"]["win_rate_ci95"] is None
    rows = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
    assert rows[1]["attempts"] == 2


def test_explicit_ai_deck_does_not_report_unused_starter_name(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(ablation, "run_game_sync", lambda **_: _result())
    assert ablation.run(_args(tmp_path, "--ai-deck-file", "ai.txt")) == 0
    summary = json.loads((next(tmp_path.iterdir()) / "summary.json").read_text())
    assert summary["ai_deck"] is None
    assert summary["ai_deck_files"] == ["ai.txt"]


def test_retry_evidence_is_preserved(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    results = iter([
        replace(_result(None, "stream_timeout"), trace=[{"event": "decision"}]),
        _result(),
    ])
    monkeypatch.setattr(ablation, "run_game_sync", lambda **_: next(results))
    assert ablation.run(_args(tmp_path, "--games", "1", "--objectives", "kl")) == 0
    traces = next(tmp_path.iterdir()) / "traces"
    assert (traces / "game_0001_attempt_01.jsonl").exists()
    assert (traces / "game_0001.jsonl").exists()


def test_rejects_empty_deck(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "load_deck_data", lambda _: {"main_deck": []})
    with pytest.raises(ValueError, match="nonempty main deck"):
        ablation.run(_args(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flag", ["--games", "--max-retries", "--max-actions"])
def test_rejects_zero_counts(tmp_path, flag):
    with pytest.raises(ValueError, match="must be positive"):
        ablation.run(_args(tmp_path, flag, "0"))


def test_rejects_objectives_for_other_pickers(tmp_path):
    with pytest.raises(ValueError, match="requires --picker kl_control"):
        ablation.run(_args(tmp_path, "--picker", "random"))


def test_benchmark_reports_latency_and_shortlist_coverage(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    result = _result()
    result.trace[:0] = [
        {"event": "decision", "decision_time_sec": 0.1,
         "picker_reasoning": {"candidate_coverage": 0.5}},
        {"event": "decision", "decision_time_sec": 0.3,
         "picker_reasoning": {"candidate_coverage": 1.0}},
    ]
    monkeypatch.setattr(ablation, "run_game_sync", lambda **_: result)
    assert ablation.run(_args(tmp_path, "--games", "1", "--objectives", "kl")) == 0
    output = next(tmp_path.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["decision_count"] == 2
    assert summary["decision_latency_sec"] == pytest.approx({"p50": 0.2, "p95": 0.29})
    assert summary["candidate_coverage_mean"] == 0.75
    assert summary["shortlisted_decisions"] == 1
