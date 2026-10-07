"""Deck evaluator — G4.

Wraps the GameRunner to measure a deck's empirical win rate against a
configurable set of reference opponents and produce per-card marginal
win-contribution estimates.

Usage::

    from src.agents.deck_builder.evaluator import DeckEvaluator, EvalResult

    ev = DeckEvaluator(
        reference_decks=[("data/decks/edh/atraxa-praetors-voice_core.txt", "heuristic")],
        games_per_pair=4,
        format="commander",
        seed=0,
    )
    result: EvalResult = asyncio.run(ev.evaluate(my_decklist, my_commander_cards))
    print(result.win_rate)
    for card, contrib in result.top_contributors(10):
        print(f"  {card:40s}  {contrib:+.3f}")
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents import make_agent
from src.integrations.decklist_loader import DecklistLoader
from src.integrations.offline_card_db import get_default_db
from src.orchestrator_legacy.game_runner import GameConfig, GameRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Win-rate and per-card marginal contribution for one deck under evaluation.

    Attributes:
        wins:          Number of games the evaluated deck won.
        losses:        Number of games the evaluated deck lost.
        total:         Total completed games.
        elapsed:       Wall-clock seconds for the full evaluation.
        contributions: Mapping ``card_name → marginal_win_contribution``.
            Positive = card appeared more often in winning games than losing.
            Scale: fractional wins attributed to the card (summed over games).
    """

    wins: int = 0
    losses: int = 0
    total: int = 0
    elapsed: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    def top_contributors(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the *n* cards with highest marginal contribution, descending."""
        return sorted(self.contributions.items(), key=lambda x: x[1], reverse=True)[:n]

    def bottom_contributors(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the *n* cards with lowest (most negative) marginal contribution."""
        return sorted(self.contributions.items(), key=lambda x: x[1])[:n]

    def __repr__(self) -> str:
        return (
            f"EvalResult(win_rate={self.win_rate:.2%}, "
            f"wins={self.wins}/{self.total}, "
            f"elapsed={self.elapsed:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DeckEvaluator:
    """Play a deck against reference opponents and measure win contribution.

    Per-card contribution is estimated via a lightweight *appearance weighting*
    approach: every card that resolved onto the battlefield (or was cast) in a
    won game receives +1/N credit (N = cards that appeared), and −1/N in a lost
    game.  This is an O(1) approximation; a full leave-one-out evaluation would
    require ``len(deck)`` × games re-runs, which is impractical for interactive
    use.

    Args:
        reference_decks:  List of ``(deck_path_str, agent_type)`` pairs.
            The deck path is loaded from disk; agent_type is any name accepted
            by ``src.agents.make_agent`` (e.g. ``"heuristic"``, ``"random"``).
        games_per_pair:   Games to play per (test_deck, reference_deck) pair.
            Total games = len(reference_decks) × games_per_pair × 2 (seat swap).
        format:           ``"commander"`` or ``"standard"``/``"modern"``.
        max_turns:        Turn limit per game (default 80).
        seed:             Base RNG seed; individual games use seed+game_idx.
        agent_type:       Agent that will pilot the deck under test (default
            ``"heuristic"`` — fast and deterministic enough for evaluation).
    """

    def __init__(
        self,
        reference_decks: list[tuple[str, str]],
        *,
        games_per_pair: int = 4,
        format: str = "commander",
        max_turns: int = 80,
        seed: int = 0,
        agent_type: str = "heuristic",
    ) -> None:
        self.reference_decks = reference_decks
        self.games_per_pair = games_per_pair
        self.format = format
        self.max_turns = max_turns
        self.seed = seed
        self.agent_type = agent_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        decklist: "Decklist",  # src.integrations.decklist_loader.Decklist
        commander_cards: list[dict[str, Any]],
        card_db: "OfflineCardDB | None" = None,  # src.integrations.offline_card_db
    ) -> EvalResult:
        """Evaluate *decklist* and return an :class:`EvalResult`."""
        t0 = time.perf_counter()

        if card_db is None:
            card_db = get_default_db()

        test_cards = self._decklist_to_cards(decklist, commander_cards, card_db)

        result = EvalResult()
        game_idx = 0

        for ref_path_str, ref_agent_type in self.reference_decks:
            ref_cards = self._load_ref_deck(ref_path_str, card_db)

            for _ in range(self.games_per_pair):
                for seat_swap in (False, True):
                    seed = self.seed + game_idx
                    game_idx += 1

                    if seat_swap:
                        p1_cards, p2_cards = ref_cards, test_cards
                        p1_agent_type, p2_agent_type = ref_agent_type, self.agent_type
                        test_is_p1 = False
                    else:
                        p1_cards, p2_cards = test_cards, ref_cards
                        p1_agent_type, p2_agent_type = self.agent_type, ref_agent_type
                        test_is_p1 = True

                    p1 = make_agent(p1_agent_type, "p1", seed=seed)
                    p2 = make_agent(p2_agent_type, "p2", seed=seed + 1000)

                    cfg = GameConfig(
                        format=self.format,
                        max_turns=self.max_turns,
                        seed=seed,
                    )
                    runner = GameRunner(
                        agents=[p1, p2],
                        decks=[p1_cards, p2_cards],
                        config=cfg,
                    )
                    game_result = await runner.run_game()

                    # Determine if test deck won.
                    winner_id = None
                    if game_result.winner is not None:
                        winner_id = getattr(game_result.winner, "player_id", None)

                    test_player_id = "p1" if test_is_p1 else "p2"
                    test_won = winner_id == test_player_id

                    result.total += 1
                    if test_won:
                        result.wins += 1
                    else:
                        result.losses += 1

                    # Credit / debit per-card contributions.
                    self._update_contributions(
                        result, game_result, test_player_id, test_won
                    )

        result.elapsed = time.perf_counter() - t0
        logger.info(
            "Evaluation done: %d/%d wins (%.1f%%) in %.1fs",
            result.wins,
            result.total,
            result.win_rate * 100,
            result.elapsed,
        )
        return result

    def evaluate_sync(
        self,
        decklist: "Decklist",
        commander_cards: list[dict[str, Any]],
        card_db: "OfflineCardDB | None" = None,
    ) -> EvalResult:
        """Synchronous wrapper around :meth:`evaluate`."""
        return asyncio.run(self.evaluate(decklist, commander_cards, card_db))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decklist_to_cards(
        self,
        decklist: "Decklist",
        commander_cards: list[dict[str, Any]],
        db: Any,
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for cmd_card in commander_cards:
            data = dict(cmd_card)
            data["is_commander"] = True
            cards.append(data)
        for name, count in decklist.mainboard.items():
            data = db.get(name)
            if data is None:
                data = _fallback_card(name)
            for _ in range(count):
                cards.append(dict(data))
        return cards

    def _load_ref_deck(self, path_str: str, db: Any) -> list[dict[str, Any]]:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Reference deck not found: %s — using fallback", path_str)
            return []
        text = path.read_text(encoding="utf-8")
        deck = DecklistLoader().from_text(text)
        cards: list[dict[str, Any]] = []
        for name in (deck.commander or []):
            data = dict(db.get(name) or _fallback_card(name))
            data["is_commander"] = True
            cards.append(data)
        for name, count in deck.mainboard.items():
            data = db.get(name) or _fallback_card(name)
            for _ in range(count):
                cards.append(dict(data))
        return cards

    def _update_contributions(
        self,
        result: EvalResult,
        game_result: Any,
        test_player_id: str,
        test_won: bool,
    ) -> None:
        """Attribute fractional win/loss credit to each card that hit the
        battlefield or was cast by the test player during this game.
        """
        # Pull the game log / state to find which cards were active.
        # The GameResult exposes the final GameState as .state.
        state = getattr(game_result, "state", None)
        if state is None:
            return

        # Collect all cards controlled or owned by the test player that
        # appeared somewhere other than the library (= were played/cast).
        seen: set[str] = set()
        for card in state.cards:
            if card.owner_id != test_player_id:
                continue
            from src.engine_legacy.game_state import Zone
            if card.zone in (Zone.LIBRARY,):
                continue  # never interacted with
            seen.add(card.name)

        if not seen:
            return

        credit = (1.0 if test_won else -1.0) / len(seen)
        for name in seen:
            result.contributions[name] = result.contributions.get(name, 0.0) + credit


def _fallback_card(name: str) -> dict:
    lower = name.lower()
    is_basic = lower in {"mountain", "island", "plains", "swamp", "forest", "wastes"}
    color = {"mountain": "R", "island": "U", "plains": "W",
             "swamp": "B", "forest": "G"}.get(lower, "C")
    return {
        "name": name,
        "type_line": f"Basic Land — {name.title()}" if is_basic else "Creature",
        "oracle_text": f"({{T}}: Add {{{color}}})" if is_basic else "",
        "mana_cost": "" if is_basic else "{2}",
        "cmc": 0 if is_basic else 2,
        "power": None if is_basic else "2",
        "toughness": None if is_basic else "2",
        "set": "STUB",
    }
