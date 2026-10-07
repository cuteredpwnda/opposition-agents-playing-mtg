from __future__ import annotations

import json
import subprocess
from argparse import Namespace

import pytest

from scripts import run_paper_pilot as pilot
from src.integrations.phase_rs import load_deck_data
from src.integrations.scryfall_bulk import file_sha256


def setup_runtime(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    submodule = root / "external" / "phase-rs"
    files = [
        submodule / "data" / "card-data.json",
        root / "data" / "scryfall" / "oracle-cards.json",
        root / "data" / "scryfall" / "rulings.json",
        root / "data" / "scryfall" / "by_name.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")
    (files[1].parent / "snapshot_manifest.json").write_text(json.dumps({
        "files": {
            path.name: {"sha256": file_sha256(path), "oracle_sha256": file_sha256(files[1])}
            for path in files[1:]
        },
    }))
    monkeypatch.setattr(pilot, "REPO_ROOT", root)
    monkeypatch.setattr(pilot, "DEFAULT_SUBMODULE", submodule)
    sources = [
        submodule / "target" / "release" / pilot._binary_name(),
        root / "scripts" / "run_paper_pilot.py",
        root / "scripts" / "run_phase_rs_ablation.py",
        root / "src" / "integrations" / "phase_rs" / "format_defaults.json",
        root / "src" / "example.py",
        root / "data" / "decks" / "modern" / "modern_mono_red_burn.txt",
        *(root / "data" / "decks" / "edh" / name for name in pilot.COMMANDER_DECKS),
    ]
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test source")

    class Server:
        uri = "ws://127.0.0.1:12345/ws"
        _adopted = False

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pilot, "PhaseServerProcess", Server)
    return Namespace(games=1, budget_seconds=30, format="Modern", seed=7,
                     output_dir=tmp_path / "outputs")


def test_pilot_interleaves_eligible_policies_and_freezes_data(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    calls = []

    def execute(cli, **kwargs):
        calls.append((cli, kwargs))
        cell = pilot.Path(cli[cli.index("--output-dir") + 1]) / "run"
        cell.mkdir(parents=True)
        (cell / "summary.json").write_text(json.dumps({"completed_games": 1}))
        return subprocess.CompletedProcess(cli, 0)

    monkeypatch.setattr(pilot.subprocess, "run", execute)
    assert pilot.run(args) == 0
    assert len(calls) == 5
    assert [cli[cli.index("--objective") + 1] for cli, _ in calls if "--objective" in cli] == [
        "kl", "efe_infogain", "efe_ambiguity",
    ]
    assert all("--ai-deck-file" in cli for cli, _ in calls)
    assert all(0 < options["timeout"] <= 30 for _, options in calls)
    output = next(args.output_dir.iterdir())
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["data_sha256"]) == 4
    assert len(manifest["server_binary_sha256"]) == 64
    assert manifest["source_sha256"]
    assert len(manifest["deck_file_sha256"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_games"] == 5 and summary["unstarted_cells"] == 0
    assert summary["input_freeze_unchanged"] is True


def test_campaign_timeout_is_not_a_draw_or_success(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)

    def timeout(cli, **kwargs):
        raise subprocess.TimeoutExpired(cli, kwargs["timeout"])

    monkeypatch.setattr(pilot.subprocess, "run", timeout)
    assert pilot.run(args) == 1
    summary = json.loads((next(args.output_dir.iterdir()) / "summary.json").read_text())
    assert summary["completed_games"] == 0
    assert summary["unstarted_cells"] == 4
    assert summary["cells"][0]["status"] == "campaign_timeout"


def test_tampered_snapshot_fails_before_server_start(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    (pilot.REPO_ROOT / "data" / "scryfall" / "oracle-cards.json").write_text("[1]")
    with pytest.raises(ValueError, match="Snapshot manifest mismatch"):
        pilot.run(args)


def test_successful_process_without_terminal_output_is_not_success(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot.subprocess, "run", lambda cli, **kwargs: subprocess.CompletedProcess(cli, 0)
    )
    assert pilot.run(args) == 1
    summary = json.loads((next(args.output_dir.iterdir()) / "summary.json").read_text())
    assert summary["completed_games"] == 0
    assert summary["incomplete_cells"] == 5
    assert all(cell["status"] == "invalid_output" for cell in summary["cells"])


@pytest.mark.parametrize("mutation", ["replace", "delete"])
@pytest.mark.parametrize("target", ["source", "deck"])
def test_input_changes_invalidate_a_completed_campaign(tmp_path, monkeypatch, mutation, target):
    args = setup_runtime(tmp_path, monkeypatch)
    relative = (
        pilot.Path("src") / "example.py" if target == "source"
        else pilot.Path("data") / "decks" / "modern" / "modern_mono_red_burn.txt"
    )

    def execute(cli, **kwargs):
        cell = pilot.Path(cli[cli.index("--output-dir") + 1]) / "run"
        cell.mkdir(parents=True)
        (cell / "summary.json").write_text(json.dumps({"completed_games": 1}))
        source = pilot.REPO_ROOT / relative
        if mutation == "replace":
            source.write_text("changed source")
        elif source.exists():
            source.unlink()
        return subprocess.CompletedProcess(cli, 0)

    monkeypatch.setattr(pilot.subprocess, "run", execute)
    assert pilot.run(args) == 1
    summary = json.loads((next(args.output_dir.iterdir()) / "summary.json").read_text())
    assert summary["completed_games"] == 5
    assert summary["input_freeze_unchanged"] is False
    assert summary["invalid_reason"] == "input_freeze_changed"
    assert summary["changed_inputs"] == [str(relative)]


def test_pilot_refuses_an_adopted_runtime(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot.PhaseServerProcess, "_adopted", True)
    with pytest.raises(ValueError, match="requires an owned server"):
        pilot.run(args)


def test_diverse_pool_contains_three_structurally_valid_decks():
    args = pilot.parse_args(["--deck-pool", "modern-diverse"])
    matchups = pilot.build_matchups(args)
    assert len(matchups) == 3
    for matchup in matchups:
        deck = load_deck_data(matchup.host)
        assert len(deck["main_deck"]) == 60
        assert not deck["commander"]
        assert matchup.opponents == (matchup.host,)


def test_round_robin_includes_mirrors_and_both_orientations():
    matchups = pilot.build_matchups(pilot.parse_args([
        "--deck-pool", "modern-diverse", "--matchups", "round-robin",
    ]))
    assert len(matchups) == 9
    pairs = {(m.host, m.opponents[0]) for m in matchups}
    assert all((opponent, host) in pairs for host, opponent in pairs)
    assert len({host for host, opponent in pairs if host == opponent}) == 3


@pytest.mark.parametrize("options", [
    ["--deck-pool", "modern-diverse"],
    ["--matchups", "round-robin"],
])
def test_commander_rejects_modern_pool_options(options):
    with pytest.raises(ValueError, match="require --format Modern"):
        pilot.build_matchups(pilot.parse_args(["--format", "Commander", *options]))


def test_custom_deck_paths_and_policy_subset_reach_each_cell(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    args.deck_files = [pilot.REPO_ROOT / pilot.MODERN_DIVERSE_DECKS[0],
                       tmp_path / "second.txt"]
    args.deck_files[1].write_text("60 Island")
    args.matchups = "round-robin"
    args.arms = ["heuristic", "random"]
    args.games = 2
    calls = []

    def execute(cli, **kwargs):
        calls.append(cli)
        cell = pilot.Path(cli[cli.index("--output-dir") + 1]) / "run"
        cell.mkdir(parents=True)
        (cell / "summary.json").write_text(json.dumps({"completed_games": 1}))
        return subprocess.CompletedProcess(cli, 0)

    monkeypatch.setattr(pilot.subprocess, "run", execute)
    assert pilot.run(args) == 0
    assert len(calls) == 16
    assert [cli[cli.index("--picker") + 1] for cli in calls[:4]] == [
        "heuristic", "random", "heuristic", "random",
    ]
    assert [cli[cli.index("--seed") + 1] for cli in calls] == ["7"] * 8 + ["8"] * 8
    output = next(args.output_dir.iterdir())
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["matchups"]) == 4
    assert len(manifest["deck_file_sha256"]) == 2
    cells = json.loads((output / "summary.json").read_text())["cells"]
    assert [cell["matchup_index"] for cell in cells[:8]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert cells[4]["matchup"]["host"] == str(args.deck_files[1])


def test_invalid_pool_fails_before_runtime(tmp_path, monkeypatch):
    args = setup_runtime(tmp_path, monkeypatch)
    args.deck_files = [tmp_path / "missing.txt"]
    with pytest.raises(ValueError, match="does not exist"):
        pilot.run(args)
    args.deck_files = [pilot.REPO_ROOT / pilot.MODERN_DIVERSE_DECKS[0]] * 2
    with pytest.raises(ValueError, match="duplicate paths"):
        pilot.run(args)


@pytest.mark.parametrize("arms", [["random", "random"], ["kl"]])
def test_commander_rejects_duplicate_or_unsupported_arms(tmp_path, monkeypatch, arms):
    args = setup_runtime(tmp_path, monkeypatch)
    args.format = "Commander"
    args.arms = arms
    with pytest.raises(ValueError, match="unique and supported"):
        pilot.run(args)
