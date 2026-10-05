import io
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_model_qualification as qualification


def test_schedule_keeps_real_components_explicit():
    rows = qualification.conditions(Path("checkpoint.pt"), ["chat-a", "chat-b"])
    assert len(rows) == 11
    lookup = dict(rows)
    assert lookup["random"] == ["--picker", "random"]
    assert lookup["heuristic"] == ["--picker", "heuristic"]
    assert lookup["chat_0"] == ["--picker", "ollama", "--ollama-model", "chat-a"]
    assert "--agent-checkpoint" in lookup["fusion"]
    larger = dict(qualification.conditions(
        Path("checkpoint.pt"), ["chat-a"], "tev1-mtg-8k:0.8b",
    ))
    assert "tev1-mtg-8k:0.8b" in larger["tev1"]
    assert "dream_search" in lookup["world_model_dream_search"]
    assert dict(zip(lookup["kl"][::2], lookup["kl"][1::2]))["--objective"] == "kl"


@pytest.mark.parametrize("only", [["unknown"], ["fusion", "fusion"]])
def test_rerun_rejects_unknown_or_duplicate_conditions(tmp_path, only):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="unique conditions"):
        qualification.run(Namespace(
            checkpoint=checkpoint, models=["chat-a"], only=only, tev1_model="tev1:0.8b",
            include_decision2=False,
            budget_seconds=10, condition_seconds=5, games=1,
        ))


def test_replicates_must_be_positive(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="games must be positive"):
        qualification.run(Namespace(
            checkpoint=checkpoint, models=["chat-a"], games=0,
            budget_seconds=10, condition_seconds=5,
        ))


def test_model_snapshot_keeps_digest_context_and_quantization(monkeypatch):
    replies = iter([
        {"models": [{"name": "tev1-mtg-8k:0.8b", "digest": "digest"}]},
        {"parameters": "num_ctx 8192", "details": {"quantization_level": "Q8_0"}},
    ])
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(next(replies)).encode()))
    snapshot = qualification.model_snapshot([
        ("tev1", ["--picker", "tev1", "--tev1-model", "tev1-mtg-8k:0.8b"]),
    ])
    assert snapshot["tev1-mtg-8k:0.8b"]["parameters"] == "num_ctx 8192"
    assert snapshot["tev1-mtg-8k:0.8b"]["digest"] == "digest"
    assert snapshot["tev1-mtg-8k:0.8b"]["details"]["quantization_level"] == "Q8_0"


def test_missing_model_snapshot_fails_before_campaign(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(b'{"models":[]}'))
    with pytest.raises(ValueError, match="not installed"):
        qualification.model_snapshot([("chat", ["--ollama-model", "missing:tag"])])


def test_native_decision2_does_not_require_an_ollama_model_snapshot(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Decision 2.0 must not query the Ollama model registry")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    assert qualification.model_snapshot([
        ("decision2", ["--picker", "decision2", "--decision2-model", "local/model"]),
    ]) == {}


@pytest.mark.parametrize("games", [1, 2])
def test_conditions_retain_failures_and_private_uri(tmp_path, monkeypatch, games):
    calls = []

    def execute(cli, **kwargs):
        calls.append(cli)
        name = Path(cli[cli.index("--output-dir") + 1]).name
        if name == "timeout":
            raise subprocess.TimeoutExpired(cli, kwargs["timeout"])
        if name == "terminal":
            out = Path(cli[cli.index("--output-dir") + 1]) / "run"
            out.mkdir()
            (out / "summary.json").write_text(json.dumps({"completed_games": games}))
        return subprocess.CompletedProcess(cli, 0)

    monkeypatch.setattr(qualification.subprocess, "run", execute)
    args = Namespace(budget_seconds=20, condition_seconds=5, seed=7,
                     games=games, ai_difficulty="Medium")
    rows = qualification.run_conditions(
        args, [(name, []) for name in ["terminal", "no-output", "timeout"]],
        tmp_path, Path("deck.txt"), "ws://localhost:12345/ws",
    )
    assert [r["status"] for r in rows] == ["finished", "invalid_output", "subprocess_timeout"]
    assert all("--autostart" not in c and "ws://localhost:12345/ws" in c for c in calls)
    assert all(c[c.index("--ai-difficulty") + 1] == "Medium" for c in calls)
    assert all(c[c.index("--games") + 1] == str(games) for c in calls)
    assert len((tmp_path / "conditions.jsonl").read_text().splitlines()) == 3
