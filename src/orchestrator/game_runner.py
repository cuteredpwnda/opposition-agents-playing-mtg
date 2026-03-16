"""
High-level game runner — sets up and runs a complete MTG game.

Coordinates agent initialization, game state setup, deck loading,
and the main game loop via LangGraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import MTGAgent
from src.engine.game_state import CardInstance, GameState, Phase, PlayerState
from src.engine.rules_engine import RulesEngine
from src.orchestrator.priority_loop import (
    advance_priority,
    get_priority_order,
    priority_action_result,
)

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for a game."""

    format: str = "commander"  # "commander" | "standard"
    starting_life: int = 40
    max_turns: int = 100


@dataclass
class GameResult:
    """Result of a completed game."""

    winner: str | None
    turns: int
    log: list[str] = field(default_factory=list)


class GameRunner:
    """Runs a complete MTG game between agents."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.engine = RulesEngine()

    async def run_game(
        self,
        agents: dict[str, MTGAgent],
        decks: dict[str, list[dict[str, Any]]],
    ) -> GameResult:
        """Run a full game to completion.

        Args:
            agents: mapping of player_id → MTGAgent
            decks: mapping of player_id → list of card dicts (from Scryfall)
        """
        game_state = self._setup_game(agents, decks)
        logger.info(f"Game started: {list(agents.keys())}")

        while not game_state.game_over and game_state.turn_number <= self.config.max_turns:
            game_state = await self._play_turn(game_state, agents)

        winner = game_state.winner if game_state.game_over else None
        result = GameResult(
            winner=winner,
            turns=game_state.turn_number,
            log=game_state.log,
        )
        logger.info(f"Game ended: winner={winner}, turns={result.turns}")
        return result

    def _setup_game(
        self,
        agents: dict[str, MTGAgent],
        decks: dict[str, list[dict[str, Any]]],
    ) -> GameState:
        """Initialize game state, shuffle libraries, draw opening hands."""
        import random

        life = self.config.starting_life
        players = {pid: PlayerState(life=life) for pid in agents}
        cards_in_zone: dict[tuple[str, str], list[CardInstance]] = {}

        for pid, deck_data in decks.items():
            library = [
                CardInstance(
                    instance_id=f"{pid}_{i}",
                    owner=pid,
                    controller=pid,
                    name=card.get("name", f"Card_{i}"),
                    oracle_text=card.get("oracle_text", ""),
                    type_line=card.get("type_line", ""),
                    mana_cost=card.get("mana_cost", ""),
                    cmc=int(card.get("cmc", 0)),
                    is_creature="Creature" in card.get("type_line", ""),
                    is_land="Land" in card.get("type_line", ""),
                    is_instant="Instant" in card.get("type_line", ""),
                    power=int(card["power"]) if card.get("power", "").isdigit() else None,
                    toughness=int(card["toughness"]) if card.get("toughness", "").isdigit() else None,
                )
                for i, card in enumerate(deck_data)
            ]
            random.shuffle(library)

            # Draw opening hand of 7
            hand = library[:7]
            library = library[7:]

            cards_in_zone[(pid, "library")] = library
            cards_in_zone[(pid, "hand")] = hand
            cards_in_zone[(pid, "battlefield")] = []
            cards_in_zone[(pid, "graveyard")] = []
            cards_in_zone[(pid, "exile")] = []
            cards_in_zone[(pid, "command")] = []

        player_ids = list(agents.keys())
        return GameState(
            players=players,
            active_player=player_ids[0],
            priority_player=player_ids[0],
            phase=Phase.UNTAP,
            turn_number=1,
            cards_in_zone=cards_in_zone,
        )

    async def _play_turn(
        self, game_state: GameState, agents: dict[str, MTGAgent]
    ) -> GameState:
        """Play a single turn (all phases)."""
        from src.engine.phases import PHASE_ORDER, advance_phase

        for phase in PHASE_ORDER:
            game_state.phase = phase

            # Give each player priority in APNAP order
            passed: set[str] = set()
            order = get_priority_order(game_state)
            game_state.priority_player = order[0]

            while True:
                pid = game_state.priority_player
                agent = agents[pid]
                legal = self.engine.get_legal_actions(game_state, pid)

                if not legal:
                    passed.add(pid)
                else:
                    action = await agent.decide_action(game_state, legal)
                    # Notify all agents of the action
                    for a in agents.values():
                        a.observe(game_state, action)

                    if action.action_type.value == "pass_priority":
                        passed.add(pid)
                    else:
                        passed.clear()
                        game_state = self.engine.execute_action(game_state, action)

                result = priority_action_result(game_state, passed)
                if result == "advance_phase":
                    break
                if result == "resolve":
                    from src.engine.stack import resolve_top
                    game_state = resolve_top(game_state)
                    passed.clear()
                    continue

                game_state = advance_priority(game_state)

            # Check SBAs after each phase
            game_state = self.engine.check_state_based_actions(game_state)
            if game_state.game_over:
                break

        if not game_state.game_over:
            game_state = advance_phase(game_state)

        return game_state
