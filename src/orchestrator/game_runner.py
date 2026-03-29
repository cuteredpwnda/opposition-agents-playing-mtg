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
from src.engine.card_database import CardDatabase
from src.engine.game_state import CardInstance, GameState, Phase, PlayerState, Zone
from src.engine.rules_engine import RulesEngine
from src.orchestrator.priority_loop import (
    advance_priority,
    get_priority_order,
    priority_action_result,
    run_priority_loop,
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

    def __init__(
        self,
        config: GameConfig | None = None,
        card_db: CardDatabase | None = None,
        self_play_collector: object | None = None,
    ):
        self.config = config or GameConfig()
        self.card_db = card_db or CardDatabase()
        self.engine = RulesEngine()
        self.self_play_collector = self_play_collector

    async def run_game(
        self,
        agents: dict[str, MTGAgent],
        decks: dict[str, list[dict[str, Any]]],
    ) -> GameResult:
        """Run a full game to completion with zero external API calls.

        All card data is pre-loaded from decks into the local CardDatabase during setup.
        The game executes deterministically with no external API dependencies after initialization.

        Args:
            agents: mapping of player_id → MTGAgent instance
            decks: mapping of player_id → list of card dicts (from Scryfall)
                   All cards in both decks must be pre-fetched before calling this method.

        Returns:
            GameResult with winner, final state, and turn log

        Note:
            After _setup_game completes, all game logic uses only locally cached card data.
            This ensures reproducible games and prevents API rate limiting during long play sessions.
        """
        game_state = self._setup_game(agents, decks)
        logger.info(f"Game started: {list(agents.keys())}")

        while not game_state.game_over and game_state.turn_number <= self.config.max_turns:
            game_state = await self._play_turn(game_state, agents)

        # Max-turn guard for non-terminal games
        if not game_state.game_over:
            game_state.game_over = True
            game_state.log("Game drawn: max turn limit reached")

        winner_name = game_state.winner.player_id if game_state.winner else None

        if self.self_play_collector is not None:
            # Finalize self-play trajectory
            winner_idx = None
            if winner_name is not None:
                player_ids = [p.player_id for p in game_state.players]
                winner_idx = player_ids.index(winner_name) if winner_name in player_ids else None
            self.self_play_collector.finish_game(winner=winner_idx, num_turns=game_state.turn_number)

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

        # PHASE 1: Populate card database (zero API calls after this point)
        # Pre-load all cards from both decks into local cache
        for pid, deck_data in decks.items():
            for card_dict in deck_data:
                card_name = card_dict.get("name", "Unknown")
                # Store card data by name for quick lookup
                # (Multiple instances of same card name reference same data)
                if not self.card_db.card_exists(card_name):
                    self.card_db.add_card(card_name, card_dict)

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

        # Load cards for each player (all data from pre-loaded database)
        all_cards: list[CardInstance] = []
        for pid, deck_data in decks.items():
            for i, card_dict in enumerate(deck_data):
                card = CardInstance(
                    instance_id=f"{pid}_{i}",
                    card_data=card_dict,  # Use pre-loaded card data
                    zone=Zone.LIBRARY,
                    owner_id=pid,
                    controller_id=pid,
                )
                all_cards.append(card)

            # Shuffle library for this player
            library_cards = [c for c in all_cards if c.owner_id == pid]
            random.shuffle(library_cards)

            # Commander: keep the first commander in command zone
            if self.config.format == "commander" and library_cards:
                commander_card = library_cards.pop(0)
                commander_card.zone = Zone.COMMAND_ZONE                # Mark the commander card instance explicitly
                commander_card.card_data["is_commander"] = True                # assign to game state commanders map
                # will be set after game_state is created
                # use player_id for map lookup
                # store temporary mapping in player object later
                # (GameState.commanders is set after GameState instantiation)
                # We'll finalize after game_state, but for now we can use state assigned below.
                # We'll use game_state.commanders by updating after state initialization.
                commander_holder = commander_card

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

        # Commander post-setup: mark each player's commander mapping
        if self.config.format == "commander":
            game_state.commanders = {
                p.player_id: next(
                    (c.instance_id for c in all_cards if c.owner_id == p.player_id and c.zone == Zone.COMMAND_ZONE),
                    "",
                )
                for p in players
            }

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

            # In untap step, creatures remove summoning sickness (gained pre-eot)
            if phase == Phase.UNTAP:
                for card in game_state.cards:
                    if card.zone == Zone.BATTLEFIELD and card.is_creature():
                        # Creatures that entered in a previous turn lose summoning sickness
                        if card.summoning_sick and card.turn_entered < game_state.turn_number:
                            card.summoning_sick = False
                        card.tapped = False  # Also untap all permanents
                
                # Reset land play allocation for all players
                for player in game_state.players:
                    player.land_plays_remaining = 1

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
            if phase == Phase.COMBAT_BEGIN:
                # Initialize combat state at the start of combat
                from src.engine.game_state import CombatState
                game_state.combat = CombatState()
            
            if phase == Phase.COMBAT_ATTACKERS:
                # Active player declares attackers with stack/priority support
                game_state.priority_player_index = game_state.active_player_index
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
                if game_state.game_over:
                    return game_state
            
            if phase == Phase.COMBAT_BLOCKERS:
                # Defending player declares blockers with stack/priority support
                defending_player_idx = (game_state.active_player_index + 1) % len(game_state.players)
                game_state.priority_player_index = defending_player_idx
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
                if game_state.game_over:
                    return game_state
            
            if phase == Phase.COMBAT_DAMAGE:
                from src.engine.combat import resolve_combat_damage
                # Resolve combat damage
                resolve_combat_damage(game_state)
            
            if phase == Phase.CLEANUP:
                # Empty all players' mana pools
                from src.engine.mana import empty_mana_pool
                for player in game_state.players:
                    empty_mana_pool(player)

            # In main phases, use full priority loop (enables stack + instant-speed)
            if is_main_phase(phase):
                # Initialize priority to active player
                game_state.priority_player_index = game_state.active_player_index
                
                # Run full priority loop until stack empties and all pass
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
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
        
        # After all phases complete, move to next turn
        game_state.turn_number += 1
        game_state.active_player_index = (game_state.active_player_index + 1) % len(game_state.players)

        return game_state
