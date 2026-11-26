"""Tests for the deck-builder evaluator (G4) and mutator (G5).

These tests run without Neo4j, JEPA checkpoints, or real card data.
They use minimal in-memory fakes to verify the evaluator/mutator
control flow and data contracts without touching the full game engine
(which would be slow in unit tests).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.deck_builder.evaluator import DeckEvaluator, EvalResult, _fallback_card
from src.agents.deck_builder.mutator import DeckMutator, MutationLog, _is_basic
from src.integrations.decklist_loader import Decklist


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

def _make_decklist(names: list[str]) -> Decklist:
    d = Decklist()
    for n in names:
        d.mainboard[n] = d.mainboard.get(n, 0) + 1
    return d


def _commander_cards() -> list[dict]:
    return [
        {
            "name": "Krenko, Mob Boss",
            "type_line": "Legendary Creature — Goblin Warrior",
            "oracle_text": "{T}: Create X 1/1 red Goblin creature tokens",
            "color_identity": ["R"],
            "cmc": 4.0,
            "is_commander": True,
        }
    ]


def _fake_db(names: list[str]) -> Any:
    """Minimal card_db stub that returns fallback cards for given names."""
    db = MagicMock()
    db.get = lambda n: (
        {"name": n, "type_line": "Creature", "oracle_text": "",
         "color_identity": ["R"], "cmc": 2.0,
         "legalities": {"commander": "legal"}}
        if n in names else None
    )
    db.all_names = lambda: iter(names)
    return db


# ---------------------------------------------------------------------------
# EvalResult helpers
# ---------------------------------------------------------------------------

def test_eval_result_win_rate_zero_games():
    r = EvalResult()
    assert r.win_rate == 0.0


def test_eval_result_win_rate():
    r = EvalResult(wins=3, total=4)
    assert r.win_rate == pytest.approx(0.75)


def test_eval_result_top_contributors():
    r = EvalResult(contributions={"A": 0.5, "B": 1.0, "C": -0.2})
    top = r.top_contributors(2)
    assert [n for n, _ in top] == ["B", "A"]


def test_eval_result_bottom_contributors():
    r = EvalResult(contributions={"A": 0.5, "B": 1.0, "C": -0.2})
    bottom = r.bottom_contributors(1)
    assert bottom[0][0] == "C"


# ---------------------------------------------------------------------------
# Fallback card helper
# ---------------------------------------------------------------------------

def test_fallback_card_mountain():
    c = _fallback_card("Mountain")
    assert "Land" in c["type_line"]
    assert c["cmc"] == 0


def test_fallback_card_creature():
    c = _fallback_card("Mystery Card")
    assert c["type_line"] == "Creature"
    assert c["cmc"] == 2


# ---------------------------------------------------------------------------
# DeckEvaluator — win/loss counting
# ---------------------------------------------------------------------------

def _build_mock_game_result(test_pid: str, test_won: bool, card_names: list[str]):
    """Build a minimal GameResult mock that the evaluator can read."""
    from src.engine.game_state import Zone

    # Fake cards owned by test player.
    fake_cards = []
    for name in card_names:
        c = MagicMock()
        c.name = name
        c.owner_id = test_pid
        c.zone = Zone.GRAVEYARD  # appeared in game (not library)
        fake_cards.append(c)
    # Add one card owned by the opponent (should be ignored).
    opp_card = MagicMock()
    opp_card.name = "Opponent Card"
    opp_card.owner_id = "p2" if test_pid == "p1" else "p1"
    opp_card.zone = Zone.GRAVEYARD
    fake_cards.append(opp_card)

    state = MagicMock()
    state.cards = fake_cards

    winner = MagicMock()
    winner.player_id = test_pid if test_won else ("p2" if test_pid == "p1" else "p1")

    result = MagicMock()
    result.state = state
    result.winner = winner
    return result


@pytest.mark.asyncio
async def test_evaluator_win_loss_counts():
    """Evaluator tallies wins and losses correctly."""
    ev = DeckEvaluator(
        reference_decks=[("fake/path.txt", "heuristic")],
        games_per_pair=1,
        format="commander",
        seed=0,
    )

    deck = _make_decklist(["Mountain"] * 99)
    commanders = _commander_cards()
    db = _fake_db(["Mountain"])

    game_results = []
    # seat 0 (test=p1) → test wins
    gr1 = _build_mock_game_result("p1", True, ["Mountain"])
    # seat 1 (test=p2) → test loses
    gr2 = _build_mock_game_result("p2", False, ["Mountain"])
    game_results = [gr1, gr2]
    call_idx = [0]

    mock_runner = AsyncMock()
    async def _run():
        r = game_results[call_idx[0] % len(game_results)]
        call_idx[0] += 1
        return r

    mock_runner.run_game = _run
    ev._load_ref_deck = lambda path, db: [_fallback_card("Island")]

    with patch("src.agents.deck_builder.evaluator.GameRunner", return_value=mock_runner):
        with patch("src.agents.deck_builder.evaluator.GameConfig"):
            with patch("src.agents.deck_builder.evaluator.make_agent", return_value=MagicMock()):
                result = await ev.evaluate(deck, commanders, db)

    assert result.total == 2
    assert result.wins == 1
    assert result.losses == 1
    assert result.win_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_evaluator_contribution_positive_on_win():
    """Cards seen in a won game get positive contribution."""
    ev = DeckEvaluator(
        reference_decks=[("fake/path.txt", "random")],
        games_per_pair=1,
        format="commander",
        seed=0,
    )
    deck = _make_decklist(["Lightning Bolt", "Mountain"])
    commanders = _commander_cards()
    db = _fake_db(["Lightning Bolt", "Mountain"])

    # One game, test wins.
    gr = _build_mock_game_result("p1", True, ["Lightning Bolt", "Mountain"])
    ev._load_ref_deck = lambda path, db: [_fallback_card("Island")]

    call_idx = [0]
    results = [gr, _build_mock_game_result("p2", False, [])]

    mock_runner = AsyncMock()
    async def _run():
        r = results[call_idx[0] % len(results)]
        call_idx[0] += 1
        return r
    mock_runner.run_game = _run

    with patch("src.agents.deck_builder.evaluator.GameRunner", return_value=mock_runner):
        with patch("src.agents.deck_builder.evaluator.GameConfig"):
            with patch("src.agents.deck_builder.evaluator.make_agent", return_value=MagicMock()):
                result = await ev.evaluate(deck, commanders, db)

    # Both cards seen in winning game → positive contribution.
    assert result.contributions.get("Lightning Bolt", 0) > 0


# ---------------------------------------------------------------------------
# _is_basic helper
# ---------------------------------------------------------------------------

def test_is_basic_true():
    assert _is_basic("Mountain")
    assert _is_basic("Island")
    assert _is_basic("Snow-Covered Forest")


def test_is_basic_false():
    assert not _is_basic("Command Tower")
    assert not _is_basic("Sol Ring")


# ---------------------------------------------------------------------------
# MutationLog
# ---------------------------------------------------------------------------

def test_mutation_log_summary_empty():
    log = MutationLog()
    s = log.summary()
    assert "0 iterations" in s


def test_mutation_log_accepted_steps():
    from src.agents.deck_builder.mutator import MutationStep
    log = MutationLog()
    log.steps = [
        MutationStep(1, [("A", "B")], 0.4, 0.5, accepted=True),
        MutationStep(2, [("C", "D")], 0.5, 0.45, accepted=False),
    ]
    assert len(log.accepted_steps()) == 1
    assert log.steps[0].delta == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# DeckMutator — early-stop when no swap candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mutator_no_swap_candidates_stops_early():
    """Mutator stops if bottom_contributors are all basics (not swappable)."""
    from src.agents.deck_builder.mutator import DeckMutator

    # Fake evaluator always returns all-basics in bottom contributors.
    fake_result = EvalResult(wins=2, total=4)
    # Only basics in contributions → all filtered out.
    fake_result.contributions = {
        "Mountain": -0.1,
        "Forest": -0.05,
        "Island": -0.08,
    }

    mock_ev = MagicMock()
    mock_ev.format = "commander"
    mock_ev.evaluate = AsyncMock(return_value=fake_result)

    db = _fake_db(["Sol Ring", "Swords to Plowshares"])
    deck = _make_decklist(["Mountain"] * 36 + ["Sol Ring"] * 10 + ["Island"] * 53)
    commanders = _commander_cards()

    mutator = DeckMutator(
        evaluator=mock_ev,
        card_db=db,
        max_iterations=5,
        swaps_per_iter=2,
        pool_sample=10,
        temperature=0.0,
        seed=0,
    )
    best, log = await mutator.mutate(deck, commanders)

    # Should have stopped after the first iteration with no non-basic candidates.
    assert len(log.steps) <= 2


@pytest.mark.asyncio
async def test_mutator_accepts_improvement():
    """Mutator accepts a swap that improves win rate."""
    from src.agents.deck_builder.mutator import DeckMutator

    call_count = [0]
    base_result = EvalResult(wins=1, total=4, contributions={"Bad Card": -0.5, "Mountain": 0.1})
    improved_result = EvalResult(wins=3, total=4, contributions={"New Card": 0.4, "Mountain": 0.1})

    async def _fake_evaluate(decklist, commanders, db=None):
        call_count[0] += 1
        # First call = baseline; subsequent = improved.
        return base_result if call_count[0] == 1 else improved_result

    mock_ev = MagicMock()
    mock_ev.format = "commander"
    mock_ev.evaluate = _fake_evaluate

    db = _fake_db(["Mountain", "Bad Card", "New Card"])
    deck = _make_decklist(["Mountain"] * 10 + ["Bad Card"] * 89)
    commanders = _commander_cards()

    mutator = DeckMutator(
        evaluator=mock_ev,
        card_db=db,
        max_iterations=2,
        swaps_per_iter=1,
        pool_sample=10,
        temperature=0.0,  # greedy
        seed=7,
    )
    best, log = await mutator.mutate(deck, commanders)

    accepted = log.accepted_steps()
    assert len(accepted) >= 1
    assert accepted[0].delta > 0
    assert log.best_win_rate > base_result.win_rate
