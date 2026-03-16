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
from src.engine.game_state import CardInstance, GameState, Phase, PlayerState, Zone
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

        winner_name = game_state.winner.player_id if game_state.winner else None
        result = GameResult(
            winner=winner_name,
            turns=game_state.turn_number,
            log=game_state.game_log,
        )
        logger.info(f"Game ended: winner={winner_name}, turns={result.turns}")
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
        from src.engine.phases import PHASE_ORDER, advance_phase, is_main_phase
        from src.engine.game_state import ActionType

        # Actually play through each phase of the turn
        phase_idx = 0
        while phase_idx < len(PHASE_ORDER):
            if game_state.game_over:
                return game_state
            
            phase = PHASE_ORDER[phase_idx]
            game_state.phase = phase
            game_state.log(f"--- {phase.value.upper()} ---")

            # In draw step, active player draws
            if phase == Phase.DRAW:
                player = game_state.active_player
                library = [c for c in game_state.cards if c.zone == Zone.LIBRARY and c.owner_id == player.player_id]
                if library:
                    card_to_draw = library[0]
                    from src.engine.zones import move_card
                    game_state = move_card(game_state, card_to_draw.instance_id, Zone.LIBRARY, Zone.HAND, player.player_id)
                    player.has_drawn_for_turn = True

            # Combat phases
            if phase == Phase.COMBAT_ATTACKERS:
                from src.engine.combat import declare_attackers
                pid = game_state.active_player.player_id
                agent = agents[pid]
                
                # Active player makes attacker decisions until they pass
                while not game_state.game_over:
                    legal = self.engine.get_legal_actions(game_state, pid)
                    
                    # Check if there are any attack options
                    attack_options = [a for a in legal if a.action_type == ActionType.DECLARE_ATTACKERS]
                    if not attack_options:
                        # No more creatures to attack with
                        break
                    
                    action = await agent.decide_action(game_state, legal)
                    
                    # Notify all observers
                    for a in agents.values():
                        await a.observe(game_state, action)
                    
                    # Check if passed
                    if action.action_type == ActionType.PASS_PRIORITY:
                        break
                    
                    # Execute attack action
                    game_state = self.engine.execute_action(game_state, action)
            
            if phase == Phase.COMBAT_BLOCKERS:
                # Defending players declare blockers (for now, empty)
                pass
            
            if phase == Phase.COMBAT_DAMAGE:
                from src.engine.combat import resolve_combat_damage
                # Resolve combat damage
                resolve_combat_damage(game_state)

            # In main phases, active player can cast spells/play lands
            if is_main_phase(phase):
                pid = game_state.active_player.player_id
                agent = agents[pid]
                
                # Player makes decisions until they pass
                while not game_state.game_over:
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
                    self.engine.check_state_based_actions(game_state)
                    if game_state.game_over:
                        return game_state
            
            # Check SBAs after each phase
            sba_events = self.engine.check_state_based_actions(game_state)
            if sba_events or game_state.game_over:
                logger.info(f"SBA after {phase.value}: {sba_events}, game_over={game_state.game_over}")
            if game_state.game_over:
                logger.info(f"Game over triggered after phase {phase.value}")
                return game_state
            
            # Move to next phase
            phase_idx += 1

        return game_state
