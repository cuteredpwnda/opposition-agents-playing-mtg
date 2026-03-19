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

from src.engine.game_state import GameState, PlayerState, Zone, Phase, CardInstance
from src.engine.game_execution import AgentGamePlayer, GameCoordinator
from src.engine.rules_engine import RulesEngine
from src.engine.agent_strategies import Strategy
from src.engine.knowledge_graph import MTGKnowledgeGraph
from src.engine.game_logger import GameLogger


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
    
    def setup_game(self, game_id: str = None) -> GameState:
        """Initialize a fresh game state.
        
        Args:
            game_id: Optional ID for the game
            
        Returns:
            Initialized GameState
        """
        from uuid import uuid4
        
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
        
        # Initialize KG if available
        if self.kg:
            self.coordinator.setup_game(self.game, self.game_id)
        
        self.logger.info(f"Players initialized", data={
            f"{player1.name}": f"{player1.life_total} HP",
            f"{player2.name}": f"{player2.life_total} HP"
        })
        
        self.phase_log.append(f"Game {self.game_id} initialized. {player1.name} ({player1.life_total}hp) vs {player2.name} ({player2.life_total}hp)")
        
        return self.game
    
    def check_win_condition(self) -> Optional[GameResult]:
        """Check if game has ended.
        
        Returns:
            GameResult or None if game continues
        """
        if not self.game:
            return None
        
        p1_life = self.game.players[0].life_total if len(self.game.players) > 0 else 0
        p2_life = self.game.players[1].life_total if len(self.game.players) > 1 else 0
        
        # Win by life total (if player dies, opponent wins)
        if p1_life <= 0 and p2_life > 0:
            return GameResult.PLAYER2_WIN
        if p2_life <= 0 and p1_life > 0:
            return GameResult.PLAYER1_WIN
        
        # Draw if max turns reached
        if self.game.turn_number >= self.max_turns:
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
            # Untap permanents
            for card in self.game.cards:
                if card.zone == Zone.BATTLEFIELD and card.tapped:
                    card.tapped = False
            actions.append("All permanents untapped")
            if self.logger:
                self.logger.debug("Permanents untapped", turn=self.game.turn_number, phase=phase.name)
        
        elif phase == Phase.UPKEEP:
            # Check upkeep triggers (placeholder)
            actions.append("Upkeep phase")
        
        elif phase == Phase.DRAW:
            # Active player draws a card
            actions.append(f"{self.game.active_player.name} draws a card")
            if self.logger:
                self.logger.action(f"Draw a card", player=self.game.active_player.player_id, turn=self.game.turn_number, phase=phase.name)
        
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
        
        elif phase == Phase.COMBAT_BEGIN:
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
            # Declare blockers (placeholder)
            actions.append("Block phase")
        
        elif phase == Phase.COMBAT_DAMAGE:
            # Damage resolution (placeholder)
            actions.append("Damage resolved")
        
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
        
        elif phase == Phase.END_STEP:
            # End-of-turn effects
            actions.append(f"{self.game.active_player.name}'s turn ends")
        
        elif phase == Phase.CLEANUP:
            # Cleanup: reset flags
            for player in self.game.players:
                player.has_drawn_for_turn = False
                player.land_plays_remaining = 1
            actions.append("Cleanup phase")
        
        self.phase_log.extend(actions)
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
                    if result == GameResult.PLAYER1_WIN:
                        winner = self.game.players[0].player_id
                        loser = self.game.players[1].player_id
                    elif result == GameResult.PLAYER2_WIN:
                        winner = self.game.players[1].player_id
                        loser = self.game.players[0].player_id
                    else:
                        winner = "Draw"
                        loser = "Draw"
                    
                    self.logger.result(f"Game Over: {result.value}", data={
                        "winner": winner,
                        "loser": loser,
                        "final_turn": self.game.turn_number,
                        f"{self.game.players[0].player_id}_hp": self.game.players[0].life_total,
                        f"{self.game.players[1].player_id}_hp": self.game.players[1].life_total
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
