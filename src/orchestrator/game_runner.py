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
    mulligan_enabled: bool = True
    max_mulligans: int = 3


@dataclass
class GameResult:
    """Result of a completed game."""

    winner: str | None
    turns: int
    game_over: bool = False
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

    def _resolve_timeout_winner(self, game_state: GameState) -> PlayerState | None:
        """Pick a winner on max-turn timeout; return None if fully tied."""
        if len(game_state.players) < 2:
            return None

        def score(player: PlayerState) -> tuple[int, int, int, int]:
            pid = player.player_id
            battlefield = sum(
                1 for c in game_state.cards
                if c.zone == Zone.BATTLEFIELD and c.controller_id == pid
            )
            hand = sum(
                1 for c in game_state.cards
                if c.zone == Zone.HAND and c.controller_id == pid
            )
            library = sum(
                1 for c in game_state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == pid
            )
            return (player.life_total, battlefield, hand, library)

        p1, p2 = game_state.players[0], game_state.players[1]
        s1, s2 = score(p1), score(p2)
        if s1 > s2:
            return p1
        if s2 > s1:
            return p2
        return None

    def _opening_hand_is_keepable(self, hand_cards: list[CardInstance]) -> bool:
        lands = sum(1 for c in hand_cards if c.is_land())
        non_lands = len(hand_cards) - lands
        return 2 <= lands <= 5 and non_lands >= 1

    def _apply_london_mulligan(
        self,
        cards: list[CardInstance],
        shuffle: bool,
        max_mulligans: int,
    ) -> int:
        import random

        mulligans_taken = 0
        while True:
            for c in cards:
                if c.zone != Zone.COMMAND_ZONE:
                    c.zone = Zone.LIBRARY

            if shuffle:
                random.shuffle(cards)

            drawables = [c for c in cards if c.zone == Zone.LIBRARY]
            hand = drawables[:7]
            for c in hand:
                c.zone = Zone.HAND

            if mulligans_taken >= max_mulligans or self._opening_hand_is_keepable(hand):
                break
            mulligans_taken += 1

        if mulligans_taken > 0:
            hand = [c for c in cards if c.zone == Zone.HAND]

            def bottom_priority(card: CardInstance) -> tuple[int, float]:
                return (0 if card.is_land() else 1, float(card.cmc or 0.0))

            to_bottom = sorted(hand, key=bottom_priority, reverse=True)[:mulligans_taken]
            for card in to_bottom:
                card.zone = Zone.LIBRARY
                cards.remove(card)
                cards.append(card)

        return mulligans_taken

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
            timeout_winner = self._resolve_timeout_winner(game_state)
            if timeout_winner is not None:
                game_state.winner = timeout_winner
                game_state.log(
                    f"Game ended: max turn limit reached, winner by tie-break = {timeout_winner.player_id}"
                )
            else:
                game_state.log("Game drawn: max turn limit reached (fully tied)")

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
            game_over=game_state.game_over,
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
            player_cards: list[CardInstance] = []
            for i, card_dict in enumerate(deck_data):
                card = CardInstance(
                    instance_id=f"{pid}_{i}",
                    card_data=card_dict,  # Use pre-loaded card data
                    zone=Zone.LIBRARY,
                    owner_id=pid,
                    controller_id=pid,
                )
                player_cards.append(card)

            # Shuffle library for this player
            random.shuffle(player_cards)

            # Commander: keep the first commander in command zone
            if self.config.format == "commander" and player_cards:
                commander_card = player_cards.pop(0)
                commander_card.zone = Zone.COMMAND_ZONE                # Mark the commander card instance explicitly
                commander_card.card_data["is_commander"] = True                # assign to game state commanders map
                # will be set after game_state is created
                # use player_id for map lookup
                # store temporary mapping in player object later
                # (GameState.commanders is set after GameState instantiation)
                # We'll finalize after game_state, but for now we can use state assigned below.
                # We'll use game_state.commanders by updating after state initialization.
                commander_holder = commander_card
                player_cards.insert(0, commander_card)

            player_state = next((p for p in players if p.player_id == pid), None)
            if self.config.mulligan_enabled:
                mulligans_taken = self._apply_london_mulligan(
                    player_cards,
                    shuffle=True,
                    max_mulligans=self.config.max_mulligans,
                )
                if player_state is not None:
                    player_state.mulligans_taken = mulligans_taken
            else:
                drawables = [c for c in player_cards if c.zone == Zone.LIBRARY]
                for card in drawables[:7]:
                    card.zone = Zone.HAND

            all_cards.extend(player_cards)

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
                else:
                    # CR 104.3c / 704.5b: drawing from empty library loses.
                    loser_idx = game_state.active_player_index
                    winner_idx = (loser_idx + 1) % len(game_state.players)
                    game_state.winner = game_state.players[winner_idx]
                    game_state.game_over = True
                    game_state.log(f"{player.player_id} attempted to draw from empty library and lost")
                    return game_state

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
                # Empty all players' mana pools and discard down to max hand size
                from src.engine.mana import empty_mana_pool
                from src.engine.zones import move_card
                for player in game_state.players:
                    empty_mana_pool(player)
                    hand_cards = [
                        c for c in game_state.cards
                        if c.zone == Zone.HAND and c.controller_id == player.player_id
                    ]
                    excess = max(0, len(hand_cards) - player.max_hand_size)
                    if excess > 0:
                        hand_cards.sort(key=lambda c: c.card_data.get("cmc", 0), reverse=True)
                        for card in hand_cards[:excess]:
                            game_state = move_card(
                                game_state,
                                card.instance_id,
                                Zone.HAND,
                                Zone.GRAVEYARD,
                                player.player_id,
                            )
                            game_state.log(f"{player.name} discards {card.name} (cleanup)")

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
