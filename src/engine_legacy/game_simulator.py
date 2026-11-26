"""
Multi-turn Game Simulator with Strategic Agent Play.

This module orchestrates complete MTG games between agents:
1. Initializes game state with starting resources
2. Progresses through game phases
3. Calls agent decision methods at appropriate points
4. Tracks game history and state changes
5. Determines victory conditions

Supports:
- Multi-turn gameplay (typically 5-15 turns)
- Agent strategy evaluation at each decision point
- Full phase progression (untap → main → combat → main2 → cleanup)
- Game termination (life total ≤ 0, deck out, concede)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.engine.game_state import GameState, PlayerState, Zone, Phase, CardInstance, TriggerType
from src.engine.game_execution import AgentGamePlayer, GameCoordinator
from src.engine.rules_engine import RulesEngine
from src.engine.agent_strategies import Strategy
from src.engine.knowledge_graph import MTGKnowledgeGraph
from src.engine.game_logger import GameLogger
from src.engine.zones import move_card
from src.engine.mana import empty_mana_pool
from src.engine import combat as combat_mod


class GameResult(str, Enum):
    """Game outcome types."""
    PLAYER1_WIN = "player1_win"
    PLAYER2_WIN = "player2_win"
    DRAW = "draw"
    ABANDONED = "abandoned"


@dataclass
class TurnAction:
    """Records a single action in a turn."""
    player_id: str
    phase: Phase
    action_type: str  # "play_card", "attack", "pass", etc.
    details: str = ""  # Card name, creature names, etc.


@dataclass
class TurnRecord:
    """Records all actions in a single turn."""
    turn_number: int
    active_player_id: str
    phase: Phase
    actions: list[TurnAction] = field(default_factory=list)
    player1_life: int = 0
    player2_life: int = 0


class GameSimulator:
    """Orchestrates multi-turn games between agents."""
    
    def __init__(self,
                 agent1: AgentGamePlayer,
                 agent2: AgentGamePlayer,
                 knowledge_graph: Optional[MTGKnowledgeGraph] = None,
                 max_turns: int = 20):
        """Initialize game simulator.
        
        Args:
            agent1: First agent (starts with priority)
            agent2: Second agent
            knowledge_graph: Optional Neo4j KG
            max_turns: Maximum turns before draw
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.kg = knowledge_graph
        self.max_turns = max_turns
        self.rules_engine = RulesEngine()
        self.coordinator = GameCoordinator(agent1, agent2, knowledge_graph, self.rules_engine)
        self.game: Optional[GameState] = None
        self.game_id: Optional[str] = None
        self.turn_history: list[TurnRecord] = []
        self.phase_log: list[str] = []
        self.logger: Optional[GameLogger] = None
    
    def setup_game(self, game_id: str = None,
                   deck1: Optional[list[dict]] = None,
                   deck2: Optional[list[dict]] = None,
                   starting_hand_size: int = 7,
                   shuffle: bool = True,
                   mulligan_enabled: bool = True,
                   max_mulligans: int = 3) -> GameState:
        """Initialize a fresh game state.

        Args:
            game_id: Optional ID for the game
            deck1: Optional list of card dicts for player 1's library.
                If omitted, a basic mono-red deck is generated.
            deck2: Optional list of card dicts for player 2's library.
            starting_hand_size: Number of cards to draw at game start (default 7)
            shuffle: Whether to shuffle libraries before drawing opening hands
            mulligan_enabled: Apply London mulligan when opening hand is 7
            max_mulligans: Maximum mulligans each player may take
        """
        from uuid import uuid4
        import random

        self.game_id = game_id or str(uuid4())
        
        # Initialize logger
        self.logger = GameLogger(self.game_id, verbose=True)
        self.logger.info(f"Game started", data={
            "player1": self.agent1.player_id,
            "player2": self.agent2.player_id,
            "strategy1": self.agent1.strategy.value,
            "strategy2": self.agent2.strategy.value,
            "max_turns": self.max_turns
        })
        
        # Create players
        player1 = PlayerState(self.agent1.player_id, f"Agent {self.agent1.strategy.value}")
        player2 = PlayerState(self.agent2.player_id, f"Agent {self.agent2.strategy.value}")
        
        # Starting resources
        player1.life_total = 20
        player2.life_total = 20
        
        # Create game
        self.game = GameState(
            game_id=self.game_id,
            players=[player1, player2],
            active_player_index=0
        )
        
        # Start at turn 1
        self.game.turn_number = 1

        # Build/shuffle libraries and draw opening hands
        deck1 = deck1 if deck1 is not None else _build_basic_red_deck()
        deck2 = deck2 if deck2 is not None else _build_basic_red_deck()
        self._load_deck(player1.player_id, deck1, shuffle=shuffle,
                        starting_hand_size=starting_hand_size,
                        mulligan_enabled=mulligan_enabled,
                        max_mulligans=max_mulligans)
        self._load_deck(player2.player_id, deck2, shuffle=shuffle,
                        starting_hand_size=starting_hand_size,
                        mulligan_enabled=mulligan_enabled,
                        max_mulligans=max_mulligans)

        # Initialize KG if available
        if self.kg:
            self.coordinator.setup_game(self.game, self.game_id)
        
        self.logger.info(f"Players initialized", data={
            f"{player1.name}": f"{player1.life_total} HP",
            f"{player2.name}": f"{player2.life_total} HP"
        })
        
        self.phase_log.append(f"Game {self.game_id} initialized. {player1.name} ({player1.life_total}hp) vs {player2.name} ({player2.life_total}hp)")
        
        return self.game

    def _load_deck(self, player_id: str, deck: list[dict],
                   shuffle: bool = True, starting_hand_size: int = 7,
                   mulligan_enabled: bool = True,
                   max_mulligans: int = 3) -> None:
        """Create CardInstances for a player's deck, shuffle, draw opening hand."""
        import random

        cards: list[CardInstance] = []
        for i, card_dict in enumerate(deck):
            cards.append(CardInstance(
                instance_id=f"{player_id}_{i}",
                card_data=dict(card_dict),
                zone=Zone.LIBRARY,
                owner_id=player_id,
                controller_id=player_id,
                tapped=False,
            ))
        if shuffle:
            random.shuffle(cards)

        # Opening hand: default simple draw for non-7 hand sizes, London
        # mulligan for normal 7-card starts when enabled.
        mulligans_taken = 0
        if starting_hand_size == 7 and mulligan_enabled:
            keep_fn, bottom_fn = self._mulligan_callbacks_for(player_id)
            mulligans_taken = self._apply_london_mulligan(
                cards,
                shuffle=shuffle,
                max_mulligans=max_mulligans,
                keep_fn=keep_fn,
                bottom_fn=bottom_fn,
            )
            player = next((p for p in self.game.players if p.player_id == player_id), None)
            if player is not None:
                player.mulligans_taken = mulligans_taken
        else:
            for c in cards[:starting_hand_size]:
                c.zone = Zone.HAND

        self.game.cards.extend(cards)

    def _mulligan_callbacks_for(self, player_id: str):
        """Return (keep_fn, bottom_fn) honouring the agent's mulligan hooks.

        Falls back to the strategy-aware default when no agent is wired up
        (kept for unit tests that drive ``_apply_london_mulligan`` with a
        bare deck list).
        """
        agent_player = None
        if getattr(self.agent1, "player_id", None) == player_id:
            agent_player = self.agent1
        elif getattr(self.agent2, "player_id", None) == player_id:
            agent_player = self.agent2

        # Prefer the underlying MTGAgent (which exposes the hooks) if the
        # AgentGamePlayer wraps one; otherwise fall through to the
        # strategy-aware default using the player's strategy attribute.
        agent_obj = getattr(agent_player, "llm_agent", None) or agent_player

        def keep_fn(hand, mulligans_taken: int, max_mulligans: int) -> bool:
            if agent_obj is not None and hasattr(agent_obj, "decide_mulligan"):
                try:
                    return bool(agent_obj.decide_mulligan(
                        hand, mulligans_taken, max_mulligans))
                except Exception:
                    pass
            from src.agents.mulligan import should_keep
            strategy = getattr(agent_player, "strategy", None)
            return should_keep(hand, strategy=strategy,
                               mulligans_taken=mulligans_taken,
                               max_mulligans=max_mulligans)

        def bottom_fn(hand, n):
            if agent_obj is not None and hasattr(agent_obj, "select_bottom_cards"):
                try:
                    return list(agent_obj.select_bottom_cards(hand, n))
                except Exception:
                    pass
            from src.agents.mulligan import select_bottom_cards
            strategy = getattr(agent_player, "strategy", None)
            return select_bottom_cards(hand, n, strategy=strategy)

        return keep_fn, bottom_fn

    def _opening_hand_is_keepable(self, hand_cards: list[CardInstance]) -> bool:
        """Simple deterministic keep heuristic for mulligans.

        Keep hands with a reasonable land count and at least one non-land spell.
        """
        lands = sum(1 for c in hand_cards if c.is_land())
        non_lands = len(hand_cards) - lands
        return 2 <= lands <= 5 and non_lands >= 1

    def _apply_london_mulligan(self,
                               cards: list[CardInstance],
                               shuffle: bool,
                               max_mulligans: int,
                               keep_fn=None,
                               bottom_fn=None) -> int:
        """Apply London mulligan to a player's deck cards in-place.

        Draw 7, optionally repeat up to max_mulligans, then put cards equal to
        mulligans taken on the bottom of the library.

        ``keep_fn(hand, mulligans_taken, max_mulligans) -> bool`` and
        ``bottom_fn(hand, n) -> list[CardInstance]`` allow agents to plug
        in their own decisions; both default to the heuristic behaviour
        used by earlier engine snapshots.
        """
        import random

        if keep_fn is None:
            def keep_fn(hand, mulligans_taken, max_mulligans):
                return self._opening_hand_is_keepable(hand)

        mulligans_taken = 0
        while True:
            # Reset all cards to library before each mulligan decision.
            for c in cards:
                c.zone = Zone.LIBRARY

            if shuffle:
                random.shuffle(cards)

            hand = cards[:7]
            for c in hand:
                c.zone = Zone.HAND

            if mulligans_taken >= max_mulligans or keep_fn(
                    hand, mulligans_taken, max_mulligans):
                break

            mulligans_taken += 1

        # Put one card on bottom per mulligan taken.
        if mulligans_taken > 0:
            hand = [c for c in cards if c.zone == Zone.HAND]
            if bottom_fn is not None:
                try:
                    to_bottom = list(bottom_fn(hand, mulligans_taken))[:mulligans_taken]
                except Exception:
                    to_bottom = []
            else:
                to_bottom = []

            if not to_bottom:
                def bottom_priority(card: CardInstance) -> tuple[int, float]:
                    # Bottom expensive non-lands first, then lands.
                    return (0 if card.is_land() else 1, float(card.cmc or 0.0))

                to_bottom = sorted(hand, key=bottom_priority, reverse=True)[:mulligans_taken]

            for card in to_bottom:
                card.zone = Zone.LIBRARY
                if card in cards:
                    cards.remove(card)
                cards.append(card)

        return mulligans_taken

    def _timeout_winner_id(self) -> Optional[str]:
        """Choose a deterministic winner when max turn limit is reached.

        Ordering is lexicographic by:
        1) life total
        2) battlefield permanents controlled
        3) cards in hand
        4) cards left in library

        Returns None when still tied after all tie-breakers.
        """
        if not self.game or len(self.game.players) < 2:
            return None

        def score(player: PlayerState) -> tuple[int, int, int, int]:
            pid = player.player_id
            battlefield = sum(
                1 for c in self.game.cards
                if c.zone == Zone.BATTLEFIELD and c.controller_id == pid
            )
            hand = sum(
                1 for c in self.game.cards
                if c.zone == Zone.HAND and c.controller_id == pid
            )
            library = sum(
                1 for c in self.game.cards
                if c.zone == Zone.LIBRARY and c.owner_id == pid
            )
            return (player.life_total, battlefield, hand, library)

        p1, p2 = self.game.players[0], self.game.players[1]
        s1, s2 = score(p1), score(p2)
        if s1 > s2:
            return p1.player_id
        if s2 > s1:
            return p2.player_id
        return None

    def check_win_condition(self) -> Optional[GameResult]:
        """Check if game has ended.
        
        Returns:
            GameResult or None if game continues
        """
        if not self.game:
            return None

        # Engine may have flagged game_over via state-based actions
        if self.game.game_over:
            winner = self.game.winner
            if winner is None:
                return GameResult.DRAW
            winner_id = getattr(winner, "player_id", winner)
            if len(self.game.players) > 0 and winner_id == self.game.players[0].player_id:
                return GameResult.PLAYER1_WIN
            if len(self.game.players) > 1 and winner_id == self.game.players[1].player_id:
                return GameResult.PLAYER2_WIN
            return GameResult.DRAW

        p1_life = self.game.players[0].life_total if len(self.game.players) > 0 else 0
        p2_life = self.game.players[1].life_total if len(self.game.players) > 1 else 0
        
        # Win by life total (if player dies, opponent wins)
        if p1_life <= 0 and p2_life > 0:
            return GameResult.PLAYER2_WIN
        if p2_life <= 0 and p1_life > 0:
            return GameResult.PLAYER1_WIN
        
        # Max-turn timeout: award winner by deterministic tie-breaker.
        if self.game.turn_number >= self.max_turns:
            winner_id = self._timeout_winner_id()
            if winner_id == self.game.players[0].player_id:
                return GameResult.PLAYER1_WIN
            if winner_id == self.game.players[1].player_id:
                return GameResult.PLAYER2_WIN
            return GameResult.DRAW
        
        return None
    
    def execute_phase(self, phase: Phase) -> list[str]:
        """Execute a single game phase.
        
        Args:
            phase: Phase to execute
            
        Returns:
            List of action descriptions
        """
        if not self.game:
            return []
        
        actions = []
        
        if phase == Phase.UNTAP:
            # Untap permanents controlled by active player; clear summoning
            # sickness for creatures that entered on a previous turn.
            from .counters import apply_stun_on_untap
            active_id = self.game.active_player.player_id
            for card in self.game.cards:
                if card.zone != Zone.BATTLEFIELD:
                    continue
                # Untap if controlled by active player or if controller is
                # unset (synthetic test cards / pre-game setup helpers).
                if card.controller_id and card.controller_id != active_id:
                    continue
                # Stun counters (CR 701.49): skip untap and remove a stun.
                if card.tapped and apply_stun_on_untap(card):
                    continue
                card.tapped = False
                if card.is_creature() and card.turn_entered < self.game.turn_number:
                    card.summoning_sick = False
            # Reset land plays for the active player
            self.game.active_player.land_plays_remaining = 1
            actions.append("All permanents untapped")
            if self.logger:
                self.logger.debug("Permanents untapped", turn=self.game.turn_number, phase=phase.name)

        elif phase == Phase.UPKEEP:
            # Fire "at the beginning of upkeep" triggers via phases hook.
            from .phases import _scan_phase_triggers
            _scan_phase_triggers(self.game, "upkeep", TriggerType.UPKEEP)
            actions.append("Upkeep phase")

        elif phase == Phase.DRAW:
            # Active player draws a card from library to hand.
            player = self.game.active_player
            library = [c for c in self.game.cards
                       if c.zone == Zone.LIBRARY and c.owner_id == player.player_id]
            if library:
                card = library[0]
                move_card(self.game, card.instance_id, Zone.LIBRARY, Zone.HAND, player.player_id)
                player.has_drawn_for_turn = True
                actions.append(f"{player.name} draws a card ({card.name})")
                if self.logger:
                    self.logger.action(f"Draw a card ({card.name})",
                                       player=player.player_id,
                                       turn=self.game.turn_number,
                                       phase=phase.name)
            else:
                # CR 104.3c / 704.5b: player who attempts to draw from an
                # empty library loses the game.
                self.game.game_over = True
                opp_idx = 1 - self.game.active_player_index
                self.game.winner = self.game.players[opp_idx]
                actions.append(f"{player.name} attempts to draw from empty library and loses")
                if self.logger:
                    self.logger.result(f"{player.name} decks out",
                                       player=player.player_id,
                                       turn=self.game.turn_number,
                                       phase=phase.name)
        
        elif phase == Phase.MAIN_1:
            # Log hand contents before playing
            if self.logger and self.game:
                active_player = self.game.active_player
                hand_cards = [c for c in self.game.cards if c.zone == Zone.HAND and c.controller_id == active_player.player_id]
                if hand_cards:
                    hand_summary = []
                    for card in hand_cards:
                        name = card.card_data.get("name", "Unknown")
                        oracle = card.card_data.get("oracle_text", "")
                        hand_summary.append(f"{name}: {oracle[:60]}...")
                    self.logger.debug(f"{active_player.name} hand ({len(hand_cards)} cards)", 
                                    turn=self.game.turn_number, phase=phase.name, 
                                    data={"cards": hand_summary})
                else:
                    self.logger.debug(f"{active_player.name} hand (empty)", 
                                    turn=self.game.turn_number, phase=phase.name)
            
            # Main phase: play cards, abilities
            phase_actions = self.coordinator.execute_main_phase_plays(self.game)
            actions.extend(phase_actions)
            for action in phase_actions:
                if self.logger and "[LLM]" in action:
                    self.logger.decision(action, turn=self.game.turn_number, phase=phase.name)
                elif self.logger:
                    self.logger.action(action, turn=self.game.turn_number, phase=phase.name)
            # Resolve everything on the stack before leaving the main phase
            self._resolve_stack_fully()

        elif phase == Phase.COMBAT_BEGIN:
            # Begin combat: initialize combat state
            combat_mod.begin_combat(self.game)
            actions.append("Combat phase begins")
            if self.logger:
                self.logger.debug("Combat begins", turn=self.game.turn_number, phase=phase.name)

        elif phase == Phase.COMBAT_ATTACKERS:
            # Declare attackers
            phase_actions = self.coordinator.execute_combat_phase(self.game)
            actions.extend(phase_actions)
            for action in phase_actions:
                if self.logger:
                    self.logger.action(action, turn=self.game.turn_number, phase=phase.name)

        elif phase == Phase.COMBAT_BLOCKERS:
            # Defending player declares blockers using a simple greedy heuristic.
            phase_actions = self._execute_block_phase()
            actions.extend(phase_actions)
            for action in phase_actions:
                if self.logger:
                    self.logger.action(action, turn=self.game.turn_number, phase=phase.name)

        elif phase == Phase.COMBAT_DAMAGE:
            # Resolve combat damage
            combat_mod.resolve_combat_damage(self.game)
            self._check_sbas()
            actions.append("Combat damage resolved")
        
        elif phase == Phase.COMBAT_END:
            actions.append("Combat phase ends")
        
        elif phase == Phase.MAIN_2:
            # Log hand contents before second main phase
            if self.logger and self.game:
                active_player = self.game.active_player
                hand_cards = [c for c in self.game.cards if c.zone == Zone.HAND and c.controller_id == active_player.player_id]
                if hand_cards:
                    hand_summary = []
                    for card in hand_cards:
                        name = card.card_data.get("name", "Unknown")
                        oracle = card.card_data.get("oracle_text", "")
                        hand_summary.append(f"{name}: {oracle[:60]}...")
                    self.logger.debug(f"{active_player.name} hand ({len(hand_cards)} cards)", 
                                    turn=self.game.turn_number, phase=phase.name, 
                                    data={"cards": hand_summary})
                else:
                    self.logger.debug(f"{active_player.name} hand (empty)", 
                                    turn=self.game.turn_number, phase=phase.name)
            
            # Second main phase
            phase_actions = self.coordinator.execute_main_phase_plays(self.game)
            actions.extend(phase_actions)
            for action in phase_actions:
                if self.logger and "[LLM]" in action:
                    self.logger.decision(action, turn=self.game.turn_number, phase=phase.name)
                elif self.logger:
                    self.logger.action(action, turn=self.game.turn_number, phase=phase.name)
            self._resolve_stack_fully()

        elif phase == Phase.END_STEP:
            # End-of-turn triggers via shared scanner.
            from .phases import _scan_phase_triggers
            _scan_phase_triggers(self.game, "end step", TriggerType.END_STEP)
            actions.append(f"{self.game.active_player.name}'s turn ends")

        elif phase == Phase.CLEANUP:
            # Cleanup: empty mana pools, enforce max hand size, reset per-turn flags
            for player in self.game.players:
                empty_mana_pool(player)
                # CR 514: discard down to max hand size during cleanup.
                hand_cards = [
                    c for c in self.game.cards
                    if c.zone == Zone.HAND and c.controller_id == player.player_id
                ]
                excess = max(0, len(hand_cards) - player.max_hand_size)
                if excess > 0:
                    # Deterministic heuristic: discard highest CMC first.
                    hand_cards.sort(key=lambda c: c.card_data.get("cmc", 0), reverse=True)
                    for card in hand_cards[:excess]:
                        move_card(self.game, card.instance_id, Zone.HAND, Zone.GRAVEYARD, player.player_id)
                        actions.append(f"{player.name} discards {card.name}")
                player.has_drawn_for_turn = False
                player.land_plays_remaining = 1
            actions.append("Cleanup phase")

        # Always check state-based actions after a phase
        self._check_sbas()
        self.phase_log.extend(actions)
        return actions

    def _resolve_stack_fully(self) -> None:
        """Resolve every item on the stack (no priority responses in this
        simplified sync simulator)."""
        # Safety bound to avoid pathological loops from cascading triggers
        guard = 0
        while self.game.stack and guard < 256:
            self.rules_engine.resolve_stack_item(self.game)
            self._check_sbas()
            if self.game.game_over:
                return
            guard += 1

    def _check_sbas(self) -> None:
        """Run state-based actions (e.g., creature death, life loss)."""
        try:
            self.rules_engine.check_state_based_actions(self.game)
        except Exception:
            # SBAs should never crash a game; log and continue
            if self.logger:
                self.logger.debug("SBA check raised", turn=self.game.turn_number)

    def _execute_block_phase(self) -> list[str]:
        """Greedy heuristic: defending player blocks each attacker with the
        first available legal blocker (one blocker per attacker, no doubles)."""
        actions: list[str] = []
        if self.game.combat is None or not self.game.combat.attackers:
            actions.append("No attackers to block")
            return actions

        defender_idx = 1 - self.game.active_player_index
        defender = self.game.players[defender_idx]
        available = [
            c for c in self.game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == defender.player_id
            and c.is_creature()
            and not c.tapped
        ]
        assignments: dict[str, list[str]] = {}
        for attacker_id in self.game.combat.attackers.keys():
            attacker = next((c for c in self.game.cards if c.instance_id == attacker_id), None)
            if attacker is None:
                continue
            blocker = next((b for b in available if combat_mod.can_block(attacker, b)), None)
            if blocker is not None:
                assignments[attacker_id] = [blocker.instance_id]
                available.remove(blocker)
                actions.append(f"{defender.name}: {blocker.name} blocks {attacker.name}")
        if assignments:
            combat_mod.declare_blockers(self.game, assignments)
        else:
            actions.append(f"{defender.name}: no blocks declared")
        return actions
    
    def advance_turn(self) -> None:
        """Move to next turn and update game state."""
        if not self.game:
            return
        
        self.game.turn_number += 1
        self.game.active_player_index = 1 - self.game.active_player_index  # Toggle player
        self.phase_log.append(f"\n=== Turn {self.game.turn_number} ===")
    
    def execute_full_turn(self) -> list[str]:
        """Execute all phases for one turn.
        
        Returns:
            Combined action log for the turn
        """
        if not self.game:
            return []
        
        all_actions = []
        phases = [
            Phase.UNTAP,
            Phase.UPKEEP,
            Phase.DRAW,
            Phase.MAIN_1,
            Phase.COMBAT_BEGIN,
            Phase.COMBAT_ATTACKERS,
            Phase.COMBAT_BLOCKERS,
            Phase.COMBAT_DAMAGE,
            Phase.COMBAT_END,
            Phase.MAIN_2,
            Phase.END_STEP,
            Phase.CLEANUP
        ]
        
        for phase in phases:
            self.game.phase = phase
            actions = self.execute_phase(phase)
            all_actions.extend(actions)
            
            # Check for immediate win (e.g., lethal)
            win = self.check_win_condition()
            if win:
                return all_actions
        
        # Update knowledge graph after each turn (if available)
        if self.kg and self.game_id:
            try:
                self.kg.build_from_game_state(self.game, self.game_id)
            except Exception as e:
                # Silently fail if KG update fails (not critical to gameplay)
                pass
        
        return all_actions
    
    def run_game(self) -> GameResult:
        """Run complete game until end condition.
        
        Returns:
            GameResult indicating winner
        """
        # Only setup if not already set up
        if not self.game:
            self.setup_game()
        
        while True:
            # Check win condition at start of turn
            result = self.check_win_condition()
            if result:
                # Log game result
                if self.logger:
                    p1_id = self.agent1.player_id
                    p2_id = self.agent2.player_id
                    life_by_id = {p.player_id: p.life_total for p in self.game.players}
                    if result == GameResult.PLAYER1_WIN:
                        winner = p1_id
                        loser = p2_id
                    elif result == GameResult.PLAYER2_WIN:
                        winner = p2_id
                        loser = p1_id
                    else:
                        winner = "Draw"
                        loser = "Draw"
                    
                    self.logger.result(f"Game Over: {result.value}", data={
                        "winner": winner,
                        "loser": loser,
                        "final_turn": self.game.turn_number,
                        f"{p1_id}_hp": life_by_id.get(p1_id),
                        f"{p2_id}_hp": life_by_id.get(p2_id),
                    })
                return result
            
            # Log turn start
            if self.logger:
                self.logger.info(f"Turn {self.game.turn_number} - {self.game.active_player.name}", turn=self.game.turn_number, data={
                    f"{self.game.players[0].name}": f"{self.game.players[0].life_total} HP",
                    f"{self.game.players[1].name}": f"{self.game.players[1].life_total} HP"
                })
            
            # Execute turn
            self.execute_full_turn()
            
            # Advance to next turn
            self.advance_turn()
            
            # Safety check
            if self.game.turn_number > self.max_turns + 5:
                if self.logger:
                    self.logger.result(f"Game exceeded max turns", data={"max_turns": self.max_turns})
                return GameResult.DRAW
    
    def get_game_summary(self) -> str:
        """Get text summary of completed game.
        
        Returns:
            Multi-line summary with winner and key stats
        """
        if not self.game:
            return "Game not initialized"
        
        summary = f"\n{'='*60}\n"
        summary += f"GAME SUMMARY: {self.game_id}\n"
        summary += f"{'='*60}\n"
        summary += f"Total Turns: {self.game.turn_number}\n"
        summary += f"Total Actions: {len(self.phase_log)}\n"
        summary += f"\nAgent 1: {self.agent1.player_id} ({self.agent1.strategy.value})\n"
        summary += f"Agent 2: {self.agent2.player_id} ({self.agent2.strategy.value})\n"
        summary += f"\nFinal Life Totals:\n"
        summary += f"  {self.game.players[0].name}: {self.game.players[0].life_total} HP\n"
        summary += f"  {self.game.players[1].name}: {self.game.players[1].life_total} HP\n"
        summary += f"\n{'='*60}\n"
        
        return summary
    
    def get_phase_log(self) -> str:
        """Get detailed phase-by-phase log.
        
        Returns:
            Multi-line log of all phases and actions
        """
        return "\n".join(self.phase_log)


# ---------------------------------------------------------------------------
# Default deck (used when callers don't supply real Scryfall data).
# Mono-red mountains + bears + bolts: enough to play actual games of MTG.
# ---------------------------------------------------------------------------

def _build_basic_red_deck() -> list[dict]:
    """Build a 60-card basic mono-red deck: 24 Mountains + 24 vanilla 2/2 Goblins
    + 12 Lightning-Bolt-style direct-damage spells.

    All entries are plain dicts shaped like the Scryfall card object so the
    rules engine can parse mana cost, type line, P/T and oracle text without
    any external lookup.
    """
    mountain = {
        "name": "Mountain",
        "mana_cost": "",
        "cmc": 0,
        "type_line": "Basic Land — Mountain",
        "oracle_text": "({T}: Add {R}.)",
    }
    goblin = {
        "name": "Goblin Recruit",
        "mana_cost": "{1}{R}",
        "cmc": 2,
        "type_line": "Creature — Goblin Warrior",
        "oracle_text": "",
        "power": "2",
        "toughness": "2",
    }
    bolt = {
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "cmc": 1,
        "type_line": "Instant",
        "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    }
    deck: list[dict] = []
    deck.extend([dict(mountain) for _ in range(24)])
    deck.extend([dict(goblin) for _ in range(24)])
    deck.extend([dict(bolt) for _ in range(12)])
    return deck
