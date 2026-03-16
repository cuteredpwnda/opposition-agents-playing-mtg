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
        from src.engine.game_state import Zone

        # Create players list
        player_ids = list(agents.keys())
        players: list[PlayerState] = []
        for pid in player_ids:
            players.append(
                PlayerState(
                    player_id=pid,
                    name=pid,
                    life_total=self.config.starting_life,
                )
            )

        # Load cards for each player
        all_cards: list[CardInstance] = []
        for pid, deck_data in decks.items():
            for i, card_dict in enumerate(deck_data):
                card = CardInstance(
                    instance_id=f"{pid}_{i}",
                    card_data=card_dict,  # Store full Scryfall card object
                    zone=Zone.LIBRARY,
                    owner_id=pid,
                    controller_id=pid,
                )
                all_cards.append(card)

            # Shuffle library for this player
            library_cards = [c for c in all_cards if c.owner_id == pid]
            random.shuffle(library_cards)

            # Draw opening hand of 7
            hand_size = 0
            for card in library_cards:
                if hand_size < 7:
                    card.zone = Zone.HAND
                    hand_size += 1
                else:
                    card.zone = Zone.LIBRARY

        # Create game state
        game_state = GameState(
            format=self.config.format,
            turn_number=1,
            active_player_index=0,
            priority_player_index=0,
            phase=Phase.UNTAP,
            players=players,
            cards=all_cards,
        )

        return game_state

    async def _play_turn(
        self, game_state: GameState, agents: dict[str, MTGAgent]
    ) -> GameState:
        """Play a single turn (all phases)."""
        from src.engine.phases import PHASE_ORDER, advance_phase
        from src.engine.game_state import ActionType

        for phase in PHASE_ORDER:
            game_state.phase = phase

            # In main phases, each player gets a chance to cast spells/play lands
            if phase in [Phase.MAIN_1, Phase.MAIN_2]:
                # Active player has priority
                pid = game_state.players[game_state.active_player_index].player_id
                agent = agents[pid]
                
                # Player makes decisions until they pass
                while True:
                    legal = self.engine.get_legal_actions(game_state, pid)
                    
                    if not legal:
                        break
                    
                    action = await agent.decide_action(game_state, legal)
                    
                    # Notify all observers of action
                    for a in agents.values():
                        await a.observe(game_state, action)
                    
                    # Check if player passed priority
                    if action.action_type == ActionType.PASS_PRIORITY:
                        break
                    
                    # Execute the action
                    game_state = self.engine.execute_action(game_state, action)
                    
                    # Check SBAs immediately
                    sba_events = self.engine.check_state_based_actions(game_state)
                    if game_state.game_over:
                        return game_state
            
            # Check SBAs after each phase
            sba_events = self.engine.check_state_based_actions(game_state)
            if game_state.game_over:
                return game_state

        # Advance to next turn
        game_state.turn_number += 1
        game_state.active_player_index = (game_state.active_player_index + 1) % len(game_state.players)
        game_state.priority_player_index = game_state.active_player_index
        game_state.phase = Phase.UNTAP

        return game_state
