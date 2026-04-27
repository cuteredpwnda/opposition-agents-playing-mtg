"""Tests for the new Phase B/D additions:
- archetype decks
- StandardBenchmarkSuite (small, all-baseline run)
- HeuristicAgent action priority
- Trajectory -> training-example pipeline
- OpponentModel.to_belief_dict
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.training.archetype_decks import (
    ARCHETYPES,
    get_archetype,
    list_archetypes,
)


def test_archetypes_have_60_cards():
    for name in list_archetypes():
        deck = get_archetype(name)
        assert len(deck) == 60, f"{name} has {len(deck)} cards"
        assert all("name" in c and "type_line" in c for c in deck)


def test_archetypes_unknown_raises():
    with pytest.raises(KeyError):
        get_archetype("not_a_real_archetype")


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------
def test_heuristic_agent_prefers_play_land_over_pass():
    from src.agents.heuristic_agent import HeuristicAgent
    from src.engine.game_state import Action, ActionType

    agent = HeuristicAgent(player_id="p1", seed=42)
    legal = [
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p1"),
        Action(action_type=ActionType.PLAY_LAND, player_id="p1"),
        Action(action_type=ActionType.CAST_SPELL, player_id="p1"),
    ]
    chosen = asyncio.run(agent.decide_action(MagicMock(), legal))
    assert chosen.action_type == ActionType.PLAY_LAND


def test_heuristic_agent_empty_returns_pass():
    from src.agents.heuristic_agent import HeuristicAgent
    from src.engine.game_state import ActionType

    agent = HeuristicAgent(player_id="p1", seed=0)
    chosen = asyncio.run(agent.decide_action(MagicMock(), []))
    assert chosen.action_type == ActionType.PASS_PRIORITY


# ---------------------------------------------------------------------------
# Neural-reasoner training pipeline (torch-free portion)
# ---------------------------------------------------------------------------
def test_trajectories_to_examples_compute_returns():
    from src.training.neural_reasoner_trainer import trajectories_to_examples
    from src.world_model.trajectory import Trajectory, Transition

    t = Trajectory(game_id="g0", winner=0)
    t.add(
        Transition(
            state_features={"a": np.array([1.0, 2.0, 3.0])},
            action_encoding=np.zeros(4),
            reward=0.0,
            done=False,
            action_type="CAST_SPELL",
        )
    )
    t.add(
        Transition(
            state_features={"a": np.array([4.0, 5.0, 6.0])},
            action_encoding=np.zeros(4),
            reward=1.0,
            done=True,
            action_type="DECLARE_ATTACKERS",
        )
    )
    examples = trajectories_to_examples([t], discount=0.9, board_features_dim=8)
    assert len(examples) == 2
    # last reward = 1, returns = [0.9, 1.0]
    assert examples[0].target_value == pytest.approx(0.9)
    assert examples[1].target_value == pytest.approx(1.0)
    assert examples[0].target_win == 1.0
    assert examples[0].board_features.shape == (8,)


def test_trajectories_to_examples_handles_unknown_winner():
    from src.training.neural_reasoner_trainer import trajectories_to_examples
    from src.world_model.trajectory import Trajectory, Transition

    t = Trajectory(game_id="g0", winner=None)
    t.add(
        Transition(
            state_features={},
            action_encoding=np.zeros(2),
            reward=0.0,
            done=True,
            action_type="PASS_PRIORITY",
        )
    )
    examples = trajectories_to_examples([t], discount=0.99, board_features_dim=4)
    assert examples[0].target_win == 0.5


# ---------------------------------------------------------------------------
# OpponentModel JSON export
# ---------------------------------------------------------------------------
def test_opponent_model_to_belief_dict(tmp_path: Path):
    from src.agents.opponent_model import CardInformation, OpponentModel

    kg = MagicMock()
    model = OpponentModel(
        opponent_id="opp",
        kg=kg,
        known_decklist={"Lightning Bolt": 4, "Mountain": 24},
    )
    model.archetype_probabilities = {"aggro": 0.7, "control": 0.1, "midrange": 0.2}
    model.predicted_hand = {"Lightning Bolt": 0.6, "Mountain": 0.4}

    snapshot = model.to_belief_dict(top_k_hand=2)
    assert snapshot["opponent_id"] == "opp"
    assert snapshot["archetype_probabilities"]["aggro"] == 0.7
    assert len(snapshot["predicted_hand_top_k"]) == 2
    assert snapshot["predicted_hand_top_k"][0]["card"] == "Lightning Bolt"
    assert any(ci["name"] == "Lightning Bolt" for ci in snapshot["card_info"])

    out = tmp_path / "belief.json"
    model.export_belief_state(out, top_k_hand=2)
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["opponent_id"] == "opp"


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------
def test_benchmark_suite_smoke(tmp_path: Path):
    """Tiny end-to-end run with two random/heuristic baselines on one
    archetype, verifying CSV/JSON outputs are written."""
    from src.training.benchmark_suite import (
        BenchmarkConfig,
        StandardBenchmarkSuite,
        default_baseline_factories,
    )

    cfg = BenchmarkConfig(
        games_per_match=1,
        max_turns=8,
        archetypes=["mono_red_aggro"],
        seed=7,
        parallel=2,
        output_dir=tmp_path / "bench",
    )
    suite = StandardBenchmarkSuite(default_baseline_factories(), cfg)
    summary = asyncio.run(suite.run())
    assert summary["total_games"] >= 1
    assert "Random" in summary["per_agent"]
    assert "Heuristic" in summary["per_agent"]
    assert (tmp_path / "bench" / "summary.json").exists()
    assert (tmp_path / "bench" / "games.csv").exists()
